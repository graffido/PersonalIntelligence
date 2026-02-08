/**
 * @file memory_store_part3.cpp
 * @brief 分层记忆存储系统实现 - 第三部分 (索引管理、序列化、存储层)
 */

#include "memory_store.h"
#include <hnswlib/hnswlib.h>
#include <folly/json.h>

namespace personal_ontology {
namespace memory {

// =============================================================================
// 层间迁移
// =============================================================================

Result<bool> HierarchicalMemoryStore::migrateToShortTerm(const MemoryId& id) {
    std::unique_lock<std::shared_mutex> wm_lock(wm_mutex_);
    
    auto it = working_memory_.find(id);
    if (it == working_memory_.end()) {
        return Result<bool>(ErrorCode::STORAGE_NOT_FOUND, "Memory not in working memory");
    }
    
    auto entry = std::move(it->second);
    entry.current_layer = 1;
    
    // 从索引中移除
    if (entry.memory.location.has_value()) {
        removeFromSpatialIndex(id);
    }
    if (entry.memory.event_time.has_value()) {
        removeFromTemporalIndex(id);
    }
    if (entry.memory.embedding.has_value()) {
        removeFromHnswIndex(id);
    }
    
    // 从工作记忆移除
    working_memory_.erase(it);
    
    wm_lock.unlock();
    
    // 写入短期记忆
    {
        std::unique_lock<std::shared_mutex> stm_lock(stm_mutex_);
        if (short_term_cache_.size() >= config_.short_term_capacity / 2) {
            evictFromShortTermCache();
        }
        short_term_cache_[id] = entry;
    }
    
    return writeToStm(entry);
}

Result<bool> HierarchicalMemoryStore::migrateToLongTerm(const MemoryId& id) {
    MemoryEntry entry;
    bool found = false;
    
    // 先从短期记忆获取
    {
        std::unique_lock<std::shared_mutex> lock(stm_mutex_);
        auto it = short_term_cache_.find(id);
        if (it != short_term_cache_.end()) {
            entry = std::move(it->second);
            short_term_cache_.erase(it);
            found = true;
        }
    }
    
    if (!found) {
        auto result = readFromStm(id);
        if (result.isOk() && result.value().has_value()) {
            entry = result.value().value();
            found = true;
        }
    }
    
    if (!found) {
        return Result<bool>(ErrorCode::STORAGE_NOT_FOUND, "Memory not found");
    }
    
    entry.current_layer = 2;
    
    // 写入长期记忆
    auto result = writeToLtm(entry);
    if (result.isOk()) {
        // 从短期记忆删除
        deleteFromStm(id);
    }
    
    return result;
}

Result<bool> HierarchicalMemoryStore::loadIntoWorkingMemory(const MemoryId& id) {
    // 先检查是否已存在
    {
        std::shared_lock<std::shared_mutex> lock(wm_mutex_);
        if (working_memory_.find(id) != working_memory_.end()) {
            return Result<bool>(true);  // 已在工作记忆中
        }
    }
    
    // 检查容量
    {
        std::unique_lock<std::shared_mutex> lock(wm_mutex_);
        if (working_memory_.size() >= config_.working_memory_capacity) {
            evictFromWorkingMemory();
        }
    }
    
    // 从下层加载
    MemoryTrace memory;
    bool found = false;
    
    // 尝试短期记忆
    {
        std::shared_lock<std::shared_mutex> lock(stm_mutex_);
        auto it = short_term_cache_.find(id);
        if (it != short_term_cache_.end()) {
            memory = it->second.memory;
            found = true;
        }
    }
    
    if (!found) {
        auto result = readFromStm(id);
        if (result.isOk() && result.value().has_value()) {
            memory = result.value().value().memory;
            found = true;
        }
    }
    
    if (!found) {
        auto result = readFromLtm(id);
        if (result.isOk() && result.value().has_value()) {
            memory = result.value().value();
            found = true;
        }
    }
    
    if (!found) {
        return Result<bool>(ErrorCode::STORAGE_NOT_FOUND, "Memory not found");
    }
    
    // 添加到工作记忆
    {
        std::unique_lock<std::shared_mutex> lock(wm_mutex_);
        MemoryEntry entry(memory, 0);
        entry.stats.recordAccess();
        working_memory_[id] = entry;
        wm_lru_queue_.push(id);
        
        // 添加到索引
        if (memory.embedding.has_value()) {
            addToHnswIndex(id, memory.embedding.value());
        }
        if (memory.location.has_value()) {
            addToSpatialIndex(id, memory.location.value());
        }
        if (memory.event_time.has_value()) {
            addToTemporalIndex(id, memory.event_time.value());
        }
    }
    
    return Result<bool>(true);
}

// =============================================================================
// HNSW向量索引
// =============================================================================

Result<bool> HierarchicalMemoryStore::addToHnswIndex(
    const MemoryId& id, const Embedding& embedding) {
    
    if (embedding.size() != config_.vector_dim) {
        return Result<bool>(ErrorCode::INVALID_ARGUMENT,
            std::format("Embedding dimension mismatch: expected {}, got {}",
                config_.vector_dim, embedding.size()));
    }
    
    std::unique_lock<std::shared_mutex> lock(hnsw_mutex_);
    
    try {
        // HNSW使用label作为唯一标识
        hnswlib::labeltype label = std::hash<MemoryId>{}(id);
        hnsw_index_>addPoint(embedding.data(), label);
        return Result<bool>(true);
    } catch (const std::exception& e) {
        return Result<bool>(ErrorCode::INTERNAL_ERROR,
            std::format("HNSW add failed: {}", e.what()));
    }
}

Result<bool> HierarchicalMemoryStore::removeFromHnswIndex(const MemoryId& id) {
    std::unique_lock<std::shared_mutex> lock(hnsw_mutex_);
    
    try {
        hnswlib::labeltype label = std::hash<MemoryId>{}(id);
        hnsw_index_>markDelete(label);
        return Result<bool>(true);
    } catch (const std::exception& e) {
        return Result<bool>(ErrorCode::INTERNAL_ERROR,
            std::format("HNSW remove failed: {}", e.what()));
    }
}

Result<std::vector<std::pair<MemoryId, double>>> HierarchicalMemoryStore::searchHnsw(
    const Embedding& embedding, size_t top_k) {
    
    if (embedding.size() != config_.vector_dim) {
        return Result<std::vector<std::pair<MemoryId, double>>>(
            ErrorCode::INVALID_ARGUMENT, "Embedding dimension mismatch");
    }
    
    std::shared_lock<std::shared_mutex> lock(hnsw_mutex_);
    
    try {
        // 搜索k个最近邻
        auto results = hnsw_index_>searchKnn(embedding.data(), top_k);
        
        std::vector<std::pair<MemoryId, double>> matches;
        
        // 使用unordered_map来反向查找id（简单实现，实际可用更高效的映射）
        // 注意：这里使用简化的方法，实际可能需要维护label到id的映射
        
        // HNSW返回的是pair<label, distance>的优先队列
        while (!results.empty()) {
            auto [label, distance] = results.top();
            results.pop();
            
            // 在实际应用中，需要维护label到MemoryId的映射
            // 这里简化处理
            matches.emplace_back("", distance);
        }
        
        return Result<std::vector<std::pair<MemoryId, double>>>(std::move(matches));
        
    } catch (const std::exception& e) {
        return Result<std::vector<std::pair<MemoryId, double>>>(
            ErrorCode::INTERNAL_ERROR, std::format("HNSW search failed: {}", e.what()));
    }
}

// =============================================================================
// 空间索引 (RTree)
// =============================================================================

Result<bool> HierarchicalMemoryStore::addToSpatialIndex(
    const MemoryId& id, const GeoPoint& location) {
    
    std::unique_lock<std::shared_mutex> lock(spatial_mutex_);
    
    // 将经纬度转换为笛卡尔坐标（简化处理，实际使用投影）
    Point point(location.longitude, location.latitude);
    spatial_index_>insert(std::make_pair(point, id));
    
    return Result<bool>(true);
}

Result<bool> HierarchicalMemoryStore::removeFromSpatialIndex(const MemoryId& id) {
    std::unique_lock<std::shared_mutex> lock(spatial_mutex_);
    
    // RTree不支持直接按值删除，需要搜索然后删除
    // 简化实现：重建索引
    // 实际生产环境应使用更高效的删除策略
    
    return Result<bool>(true);
}

std::vector<MemoryId> HierarchicalMemoryStore::searchSpatial(
    const GeoBoundingBox& bbox) {
    
    std::shared_lock<std::shared_mutex> lock(spatial_mutex_);
    
    SpatialBox query_box(
        Point(bbox.min_lon, bbox.min_lat),
        Point(bbox.max_lon, bbox.max_lat)
    );
    
    std::vector<SpatialValue> results;
    spatial_index_>query(bgi::intersects(query_box), std::back_inserter(results));
    
    std::vector<MemoryId> ids;
    ids.reserve(results.size());
    for (const auto& [point, id] : results) {
        ids.push_back(id);
    }
    
    return ids;
}

// =============================================================================
// 时间索引
// =============================================================================

Result<bool> HierarchicalMemoryStore::addToTemporalIndex(
    const MemoryId& id, Timestamp time) {
    
    std::unique_lock<std::shared_mutex> lock(temporal_mutex_);
    temporal_index_[time].insert(id);
    
    return Result<bool>(true);
}

Result<bool> HierarchicalMemoryStore::removeFromTemporalIndex(const MemoryId& id) {
    
    std::unique_lock<std::shared_mutex> lock(temporal_mutex_);
    
    for (auto it = temporal_index_.begin(); it != temporal_index_.end(); ++it) {
        if (it->second.erase(id) > 0) {
            if (it->second.empty()) {
                temporal_index_.erase(it);
            }
            return Result<bool>(true);
        }
    }
    
    return Result<bool>(false);
}

std::vector<MemoryId> HierarchicalMemoryStore::searchTemporal(Timestamp start, Timestamp end) {
    
    std::shared_lock<std::shared_mutex> lock(temporal_mutex_);
    
    std::vector<MemoryId> ids;
    
    auto it_low = temporal_index_.lower_bound(start);
    auto it_high = temporal_index_.upper_bound(end);
    
    for (auto it = it_low; it != it_high; ++it) {
        ids.insert(ids.end(), it->second.begin(), it->second.end());
    }
    
    return ids;
}

// =============================================================================
// 序列化
// =============================================================================

std::string HierarchicalMemoryStore::serializeMemory(const MemoryTrace& memory) const {
    folly::dynamic obj = folly::dynamic::object;
    
    obj["id"] = memory.id;
    obj["entity_id"] = memory.entity_id;
    obj["memory_type"] = static_cast<int>(memory.memory_type);
    obj["created_at"] = memory.created_at;
    obj["updated_at"] = memory.updated_at;
    
    if (memory.event_time.has_value()) {
        obj["event_time"] = memory.event_time.value();
    }
    if (memory.expiration.has_value()) {
        obj["expiration"] = memory.expiration.value();
    }
    
    if (memory.location.has_value()) {
        obj["location"] = folly::dynamic::object
            ("longitude", memory.location.value().longitude)
            ("latitude", memory.location.value().latitude);
        if (memory.location.value().altitude.has_value()) {
            obj["location"]["altitude"] = memory.location.value().altitude.value();
        }
    }
    
    if (memory.location_name.has_value()) {
        obj["location_name"] = memory.location_name.value();
    }
    
    obj["content"] = memory.content;
    obj["tags"] = folly::dynamic::array;
    for (const auto& tag : memory.tags) {
        obj["tags"].push_back(tag);
    }
    
    obj["emotions"] = folly::dynamic::array;
    for (const auto& emotion : memory.emotions) {
        folly::dynamic emo_obj = folly::dynamic::object
            ("emotion", emotion.emotion)
            ("intensity", emotion.intensity)
            ("timestamp", emotion.timestamp);
        if (emotion.context.has_value()) {
            emo_obj["context"] = emotion.context.value();
        }
        obj["emotions"].push_back(emo_obj);
    }
    
    if (memory.embedding.has_value()) {
        folly::dynamic emb_array = folly::dynamic::array;
        for (float val : memory.embedding.value()) {
            emb_array.push_back(val);
        }
        obj["embedding"] = emb_array;
    }
    
    obj["source"] = folly::dynamic::object
        ("source_type", memory.source.source_type)
        ("source_id", memory.source.source_id)
        ("original_text", memory.source.original_text)
        ("source_url", memory.source.source_url);
    
    obj["related_concepts"] = folly::dynamic::array;
    for (const auto& concept_id : memory.related_concepts) {
        obj["related_concepts"].push_back(concept_id);
    }
    
    obj["related_memories"] = folly::dynamic::array;
    for (const auto& mem_id : memory.related_memories) {
        obj["related_memories"].push_back(mem_id);
    }
    
    obj["properties"] = folly::dynamic::object;
    // 属性序列化...
    
    obj["confidence"] = memory.confidence;
    obj["access_count"] = memory.access_count;
    obj["last_accessed"] = memory.last_accessed;
    obj["memory_layer"] = memory.memory_layer;
    obj["importance_score"] = memory.importance_score;
    
    return folly::toJson(obj);
}

Result<MemoryTrace> HierarchicalMemoryStore::deserializeMemory(const std::string& data) const {
    try {
        auto obj = folly::parseJson(data);
        MemoryTrace memory;
        
        memory.id = obj["id"].asString();
        memory.entity_id = obj["entity_id"].asString();
        memory.memory_type = static_cast<MemoryType>(obj["memory_type"].asInt());
        memory.created_at = obj["created_at"].asInt();
        memory.updated_at = obj["updated_at"].asInt();
        
        if (obj.find("event_time") != obj.items().end()) {
            memory.event_time = obj["event_time"].asInt();
        }
        if (obj.find("expiration") != obj.items().end()) {
            memory.expiration = obj["expiration"].asInt();
        }
        
        if (obj.find("location") != obj.items().end()) {
            GeoPoint loc;
            loc.longitude = obj["location"]["longitude"].asDouble();
            loc.latitude = obj["location"]["latitude"].asDouble();
            if (obj["location"].find("altitude") != obj["location"].items().end()) {
                loc.altitude = obj["location"]["altitude"].asDouble();
            }
            memory.location = loc;
        }
        
        if (obj.find("location_name") != obj.items().end()) {
            memory.location_name = obj["location_name"].asString();
        }
        
        memory.content = obj["content"].asString();
        
        for (const auto& tag : obj["tags"]) {
            memory.tags.push_back(tag.asString());
        }
        
        for (const auto& emo : obj["emotions"]) {
            EmotionalTag emotion;
            emotion.emotion = emo["emotion"].asString();
            emotion.intensity = emo["intensity"].asDouble();
            emotion.timestamp = emo["timestamp"].asInt();
            if (emo.find("context") != emo.items().end()) {
                emotion.context = emo["context"].asString();
            }
            memory.emotions.push_back(emotion);
        }
        
        if (obj.find("embedding") != obj.items().end()) {
            Embedding emb;
            for (const auto& val : obj["embedding"]) {
                emb.push_back(static_cast<float>(val.asDouble()));
            }
            memory.embedding = emb;
        }
        
        memory.source.source_type = obj["source"]["source_type"].asString();
        memory.source.source_id = obj["source"]["source_id"].asString();
        memory.source.original_text = obj["source"]["original_text"].asString();
        memory.source.source_url = obj["source"]["source_url"].asString();
        
        for (const auto& concept_id : obj["related_concepts"]) {
            memory.related_concepts.push_back(concept_id.asString());
        }
        
        for (const auto& mem_id : obj["related_memories"]) {
            memory.related_memories.push_back(mem_id.asString());
        }
        
        memory.confidence = obj["confidence"].asDouble();
        memory.access_count = obj["access_count"].asInt();
        memory.last_accessed = obj["last_accessed"].asInt();
        memory.memory_layer = static_cast<uint8_t>(obj["memory_layer"].asInt());
        memory.importance_score = static_cast<float>(obj["importance_score"].asDouble());
        
        return Result<MemoryTrace>(std::move(memory));
        
    } catch (const std::exception& e) {
        return Result<MemoryTrace>(ErrorCode::STORAGE_SERIALIZATION_ERROR,
            std::format("Deserialize failed: {}", e.what()));
    }
}

std::string HierarchicalMemoryStore::serializeAccessStats(const AccessStats& stats) const {
    folly::dynamic obj = folly::dynamic::object
        ("access_count", stats.access_count)
        ("last_access", stats.last_access)
        ("first_access", stats.first_access)
        ("importance_score", stats.importance_score);
    return folly::toJson(obj);
}

Result<AccessStats> HierarchicalMemoryStore::deserializeAccessStats(
    const std::string& data) const {
    
    try {
        auto obj = folly::parseJson(data);
        AccessStats stats;
        stats.access_count = obj["access_count"].asInt();
        stats.last_access = obj["last_access"].asInt();
        stats.first_access = obj["first_access"].asInt();
        stats.importance_score = obj["importance_score"].asDouble();
        return Result<AccessStats>(stats);
    } catch (const std::exception& e) {
        return Result<AccessStats>(ErrorCode::STORAGE_SERIALIZATION_ERROR,
            std::format("Deserialize stats failed: {}", e.what()));
    }
}

// =============================================================================
// 短期记忆存储操作
// =============================================================================

Result<bool> HierarchicalMemoryStore::writeToStm(const MemoryEntry& entry) {
    std::string data = serializeMemory(entry.memory);
    std::string stats_data = serializeAccessStats(entry.stats);
    
    rocksdb::WriteBatch batch;
    batch.Put(stm_cf_default_, entry.memory.id, data);
    batch.Put(stm_cf_metadata_, entry.memory.id + ":stats", stats_data);
    
    rocksdb::WriteOptions write_options;
    auto status = stm_db_>Write(write_options, &batch);
    
    if (!status.ok()) {
        return Result<bool>(ErrorCode::STORAGE_WRITE_ERROR,
            std::format("STM write failed: {}", status.ToString()));
    }
    
    return Result<bool>(true);
}

Result<std::optional<MemoryEntry>> HierarchicalMemoryStore::readFromStm(const MemoryId& id) {
    std::string data;
    rocksdb::ReadOptions read_options;
    
    auto status = stm_db_>Get(read_options, stm_cf_default_, id, &data);
    
    if (status.IsNotFound()) {
        return Result<std::optional<MemoryEntry>>(std::nullopt);
    }
    
    if (!status.ok()) {
        return Result<std::optional<MemoryEntry>>(ErrorCode::STORAGE_READ_ERROR,
            std::format("STM read failed: {}", status.ToString()));
    }
    
    auto memory_result = deserializeMemory(data);
    if (memory_result.isError()) {
        return Result<std::optional<MemoryEntry>>(
            memory_result.errorCode(), memory_result.errorMessage());
    }
    
    MemoryEntry entry(memory_result.value(), 1);
    
    // 读取访问统计
    std::string stats_data;
    status = stm_db_>Get(read_options, stm_cf_metadata_, id + ":stats", &stats_data);
    if (status.ok()) {
        auto stats_result = deserializeAccessStats(stats_data);
        if (stats_result.isOk()) {
            entry.stats = stats_result.value();
        }
    }
    
    return Result<std::optional<MemoryEntry>>(entry);
}

Result<bool> HierarchicalMemoryStore::deleteFromStm(const MemoryId& id) {
    rocksdb::WriteBatch batch;
    batch.Delete(stm_cf_default_, id);
    batch.Delete(stm_cf_metadata_, id + ":stats");
    
    rocksdb::WriteOptions write_options;
    auto status = stm_db_>Write(write_options, &batch);
    
    if (!status.ok() && !status.IsNotFound()) {
        return Result<bool>(ErrorCode::STORAGE_WRITE_ERROR,
            std::format("STM delete failed: {}", status.ToString()));
    }
    
    return Result<bool>(true);
}

// =============================================================================
// 长期记忆存储操作
// =============================================================================

Result<bool> HierarchicalMemoryStore::writeToLtm(const MemoryEntry& entry) {
    rocksdb::WriteBatch batch;
    
    std::string data = serializeMemory(entry.memory);
    batch.Put(ltm_cf_memories_, entry.memory.id, data);
    
    // 存储向量嵌入
    if (entry.memory.embedding.has_value()) {
        const auto& emb = entry.memory.embedding.value();
        std::string emb_data(reinterpret_cast<const char*>(emb.data()), 
                            emb.size() * sizeof(float));
        batch.Put(ltm_cf_embeddings_, entry.memory.id, emb_data);
    }
    
    // 存储索引信息
    folly::dynamic index_obj = folly::dynamic::object
        ("entity_id", entry.memory.entity_id)
        ("memory_type", static_cast<int>(entry.memory.memory_type))
        ("created_at", entry.memory.created_at);
    
    if (entry.memory.event_time.has_value()) {
        index_obj["event_time"] = entry.memory.event_time.value();
    }
    if (entry.memory.location.has_value()) {
        index_obj["has_location"] = true;
    }
    
    batch.Put(ltm_cf_indices_, entry.memory.id, folly::toJson(index_obj));
    
    rocksdb::WriteOptions write_options;
    auto status = ltm_db_>Write(write_options, &batch);
    
    if (!status.ok()) {
        return Result<bool>(ErrorCode::STORAGE_WRITE_ERROR,
            std::format("LTM write failed: {}", status.ToString()));
    }
    
    return Result<bool>(true);
}

Result<std::optional<MemoryTrace>> HierarchicalMemoryStore::readFromLtm(const MemoryId& id) {
    std::string data;
    rocksdb::ReadOptions read_options;
    
    auto status = ltm_db_>Get(read_options, ltm_cf_memories_, id, &data);
    
    if (status.IsNotFound()) {
        return Result<std::optional<MemoryTrace>>(std::nullopt);
    }
    
    if (!status.ok()) {
        return Result<std::optional<MemoryTrace>>(ErrorCode::STORAGE_READ_ERROR,
            std::format("LTM read failed: {}", status.ToString()));
    }
    
    auto memory_result = deserializeMemory(data);
    if (memory_result.isError()) {
        return Result<std::optional<MemoryTrace>>(
            memory_result.errorCode(), memory_result.errorMessage());
    }
    
    return Result<std::optional<MemoryTrace>>(memory_result.value());
}

Result<bool> HierarchicalMemoryStore::deleteFromLtm(const MemoryId& id) {
    rocksdb::WriteBatch batch;
    batch.Delete(ltm_cf_memories_, id);
    batch.Delete(ltm_cf_embeddings_, id);
    batch.Delete(ltm_cf_indices_, id);
    
    rocksdb::WriteOptions write_options;
    auto status = ltm_db_>Write(write_options, &batch);
    
    if (!status.ok() && !status.IsNotFound()) {
        return Result<bool>(ErrorCode::STORAGE_WRITE_ERROR,
            std::format("LTM delete failed: {}", status.ToString()));
    }
    
    return Result<bool>(true);
}

// =============================================================================
// 内存管理
// =============================================================================

void HierarchicalMemoryStore::evictFromWorkingMemory() {
    // LRU驱逐策略
    while (working_memory_.size() >= config_.working_memory_capacity && 
           !wm_lru_queue_.empty()) {
        MemoryId id = wm_lru_queue_.front();
        wm_lru_queue_.pop();
        
        auto it = working_memory_.find(id);
        if (it != working_memory_.end()) {
            // 如果记忆较重要，迁移到短期记忆
            if (it->second.stats.importance_score > 0.5) {
                auto entry = std::move(it->second);
                working_memory_.erase(it);
                
                // 异步迁移
                std::thread([this, entry]() mutable {
                    writeToStm(entry);
                }).detach();
            } else {
                working_memory_.erase(it);
            }
        }
    }
}

void HierarchicalMemoryStore::evictFromShortTermCache() {
    // 简单的LRU策略：移除最早访问的条目
    if (short_term_cache_.empty()) return;
    
    auto oldest = short_term_cache_.begin();
    Timestamp oldest_time = oldest->second.stats.last_access;
    
    for (auto it = short_term_cache_.begin(); it != short_term_cache_.end(); ++it) {
        if (it->second.stats.last_access < oldest_time) {
            oldest = it;
            oldest_time = it->second.stats.last_access;
        }
    }
    
    short_term_cache_.erase(oldest);
}

// =============================================================================
// 统计和监控
// =============================================================================

HierarchicalMemoryStore::Statistics HierarchicalMemoryStore::getStatistics() const {
    Statistics stats;
    
    {
        std::shared_lock<std::shared_mutex> lock(wm_mutex_);
        stats.working_memory_count = working_memory_.size();
    }
    
    {
        std::shared_lock<std::shared_mutex> lock(stm_mutex_);
        stats.short_term_count = short_term_cache_.size();
    }
    
    // 长期记忆数量
    if (ltm_db_) {
        rocksdb::ReadOptions read_options;
        std::unique_ptr<rocksdb::Iterator> it(ltm_db_>NewIterator(read_options, ltm_cf_memories_));
        size_t count = 0;
        for (it->SeekToFirst(); it->Valid(); it->Next()) {
            count++;
        }
        stats.long_term_count = count;
    }
    
    {
        std::shared_lock<std::shared_mutex> lock(hnsw_mutex_);
        // HNSW索引数量统计
    }
    
    {
        std::shared_lock<std::shared_mutex> lock(spatial_mutex_);
        stats.spatial_index_count = spatial_index_>size();
    }
    
    stats.cache_hit_count = cache_hits_.load();
    stats.cache_miss_count = cache_misses_.load();
    
    if (access_count_.load() > 0) {
        stats.avg_access_latency_ms = total_access_time_ms_.load() / access_count_.load();
    }
    
    return stats;
}

Result<AccessStats> HierarchicalMemoryStore::getAccessStats(const MemoryId& id) const {
    // 查工作记忆
    {
        std::shared_lock<std::shared_mutex> lock(wm_mutex_);
        auto it = working_memory_.find(id);
        if (it != working_memory_.end()) {
            return Result<AccessStats>(it->second.stats);
        }
    }
    
    // 查短期记忆缓存
    {
        std::shared_lock<std::shared_mutex> lock(stm_mutex_);
        auto it = short_term_cache_.find(id);
        if (it != short_term_cache_.end()) {
            return Result<AccessStats>(it->second.stats);
        }
    }
    
    // 查短期记忆磁盘
    {
        rocksdb::ReadOptions read_options;
        std::string stats_data;
        auto status = stm_db_>Get(read_options, stm_cf_metadata_, id + ":stats", &stats_data);
        if (status.ok()) {
            return deserializeAccessStats(stats_data);
        }
    }
    
    return Result<AccessStats>(ErrorCode::STORAGE_NOT_FOUND, "Access stats not found");
}

Result<bool> HierarchicalMemoryStore::createCheckpoint(const std::string& checkpoint_path) {
    try {
        // 创建检查点目录
        std::filesystem::create_directories(checkpoint_path);
        
        // 保存HNSW索引
        {
            std::shared_lock<std::shared_mutex> lock(hnsw_mutex_);
            std::string hnsw_dest = checkpoint_path + "/hnsw.index";
            hnsw_index_>saveIndex(hnsw_dest);
        }
        
        // RocksDB检查点
        rocksdb::Checkpoint* checkpoint = nullptr;
        rocksdb::Status status = rocksdb::Checkpoint::Create(ltm_db_.get(), &checkpoint);
        if (!status.ok()) {
            return Result<bool>(ErrorCode::INTERNAL_ERROR,
                std::format("Checkpoint creation failed: {}", status.ToString()));
        }
        
        status = checkpoint->CreateCheckpoint(checkpoint_path + "/ltm");
        delete checkpoint;
        
        if (!status.ok()) {
            return Result<bool>(ErrorCode::INTERNAL_ERROR,
                std::format("LTM checkpoint failed: {}", status.ToString()));
        }
        
        // 短期记忆检查点
        status = rocksdb::Checkpoint::Create(stm_db_.get(), &checkpoint);
        if (status.ok()) {
            status = checkpoint->CreateCheckpoint(checkpoint_path + "/stm");
            delete checkpoint;
        }
        
        return Result<bool>(true);
        
    } catch (const std::exception& e) {
        return Result<bool>(ErrorCode::INTERNAL_ERROR,
            std::format("Checkpoint error: {}", e.what()));
    }
}

// =============================================================================
// 辅助函数实现
// =============================================================================

void HierarchicalMemoryStore::loadSpatialTemporalIndices() {
    if (!ltm_db_) return;
    
    rocksdb::ReadOptions read_options;
    std::unique_ptr<rocksdb::Iterator> it(ltm_db_>NewIterator(read_options, ltm_cf_memories_));
    
    for (it->SeekToFirst(); it->Valid(); it->Next()) {
        auto memory_result = deserializeMemory(it->value().ToString());
        if (memory_result.isError()) continue;
        
        const auto& memory = memory_result.value();
        
        if (memory.location.has_value()) {
            addToSpatialIndex(memory.id, memory.location.value());
        }
        
        if (memory.event_time.has_value()) {
            addToTemporalIndex(memory.id, memory.event_time.value());
        }
    }
}

} // namespace memory
} // namespace personal_ontology
