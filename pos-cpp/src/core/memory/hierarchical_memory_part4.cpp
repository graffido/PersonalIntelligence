/**
 * @file hierarchical_memory.cpp (Part 4)
 * @brief 四层分层记忆存储系统实现 - Part 4
 * 层间转换、维护任务、序列化与统计
 */

#include "hierarchical_memory.h"
#include <folly/json.h>
#include <chrono>
#include <fstream>
#include <filesystem>
#include <zip.h>  // 用于检查点压缩

namespace personal_ontology {
namespace memory {

// =============================================================================
// 层间转换: 工作 → 长期记忆 (巩固)
// =============================================================================

Result<MemoryId> QuadLayerMemoryStore::consolidate(const MemoryId& working_id) {
    // 1. 从工作记忆获取
    MemoryTrace memory;
    {
        std::shared_lock<std::shared_mutex> lock(working_mutex_);
        auto it = working_memory_.find(working_id);
        if (it == working_memory_.end()) {
            return Result<MemoryId>(ErrorCode::STORAGE_NOT_FOUND,
                std::format("Memory not in working memory: {}", working_id));
        }
        memory = it->second.memory;
    }
    
    // 2. 存储到长期记忆
    auto result = storeLongTerm(memory);
    if (result.isError()) {
        return result;
    }
    
    // 3. 从工作记忆移除
    {
        std::unique_lock<std::shared_mutex> lock(working_mutex_);
        working_memory_.erase(working_id);
        if (focused_id_ == working_id) {
            focused_id_ = std::nullopt;
        }
    }
    
    return result;
}

void QuadLayerMemoryStore::consolidateWorkingToLTM() {
    std::vector<MemoryId> to_consolidate;
    
    {
        std::shared_lock<std::shared_mutex> lock(working_mutex_);
        auto now = MemoryTrace::now();
        
        for (const auto& [id, entry] : working_memory_) {
            // 选择需要巩固的条目
            // 1. 复述次数多
            // 2. 激活水平高但不在焦点
            // 3. 存在时间较长
            bool should_consolidate = 
                entry.rehearsal_count >= 3 ||
                (entry.activation_level > 0.7f && !entry.is_focused) ||
                (now - entry.timestamp > config_.working.max_age_ms / 2);
            
            if (should_consolidate) {
                to_consolidate.push_back(id);
            }
        }
    }
    
    // 执行巩固
    for (const auto& id : to_consolidate) {
        consolidate(id);
    }
}

// =============================================================================
// 层间转换: 长期 → 工作记忆 (回忆)
// =============================================================================

Result<bool> QuadLayerMemoryStore::recallToWorking(const MemoryId& ltm_id) {
    // 1. 从长期记忆检索
    auto memory_result = retrieveLongTerm(ltm_id);
    if (memory_result.isError()) {
        return Result<bool>(memory_result.errorCode(), memory_result.errorMessage());
    }
    
    // 2. 加载到工作记忆
    auto load_result = loadToWorkingMemory(memory_result.value(), true);
    if (load_result.isError()) {
        return load_result;
    }
    
    return Result<bool>(true);
}

// =============================================================================
// 层间转换: 长期 → 参数记忆 (学习)
// =============================================================================

Result<bool> QuadLayerMemoryStore::learnFromExperiences() {
    std::unique_lock<std::shared_mutex> lock(ltm_mutex_);
    
    // 从访问统计中学习用户偏好
    // 这里简化处理，实际应该使用更复杂的机器学习算法
    
    // 1. 分析访问频率高的记忆类型
    std::unordered_map<std::string, uint32_t> type_counts;
    
    std::unique_ptr<rocksdb::Iterator> it(ltm_db_>NewIterator(rocksdb::ReadOptions(), ltm_cf_memories_));
    for (it->SeekToFirst(); it->Valid() && type_counts.size() < 1000; it->Next()) {
        auto memory_result = deserializeMemory(it->value().ToString());
        if (memory_result.isOk()) {
            type_counts[memory_result.value().type]++;
        }
    }
    
    lock.unlock();
    
    // 2. 更新类别权重
    if (!type_counts.empty()) {
        uint32_t total = 0;
        for (const auto& [type, count] : type_counts) {
            total += count;
        }
        
        access_pattern_.category_weights.clear();
        for (const auto& [type, count] : type_counts) {
            access_pattern_.category_weights.push_back(static_cast<float>(count) / total);
        }
    }
    
    return Result<bool>(true);
}

// =============================================================================
// 维护任务
// =============================================================================

void QuadLayerMemoryStore::maintenanceLoop() {
    while (!stop_maintenance_) {
        std::unique_lock<std::mutex> lock(maintenance_mutex_);
        maintenance_cv_.wait_for(lock, 
            std::chrono::milliseconds(config_.maintenance_interval_ms),
            [&]() { return stop_maintenance_.load(); });
        
        if (stop_maintenance_) break;
        
        lock.unlock();
        
        // 执行维护任务
        runMaintenance();
    }
}

Result<MemoryMaintenanceResult> QuadLayerMemoryStore::runMaintenance() {
    auto start = std::chrono::high_resolution_clock::now();
    MemoryMaintenanceResult result;
    
    // 1. 衰减感知记忆
    size_t before = sensory_buffer_.size();
    decaySensoryMemory();
    result.decayed_sensory = before - sensory_buffer_.size();
    
    // 2. 衰减工作记忆
    before = working_memory_.size();
    decayWorkingMemory();
    result.decayed_working = before - working_memory_.size();
    
    // 3. 巩固重要记忆
    size_t ltm_before = getStatistics().long_term_count;
    consolidateWorkingToLTM();
    result.consolidated = getStatistics().long_term_count - ltm_before;
    
    // 4. 遗忘不重要的长期记忆
    if (config_.enable_forgetting) {
        result.forgotten = cleanupExpiredRelations();
    }
    
    // 5. 更新访问模式
    updateAccessPatternModel();
    
    // 6. 从经验学习
    if (config_.parameter.enable_online_learning) {
        auto learn_result = learnFromExperiences();
        if (learn_result.isOk()) {
            result.learned_params = 1;
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    result.duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        end - start).count();
    
    return Result<MemoryMaintenanceResult>(result);
}

size_t QuadLayerMemoryStore::cleanupExpiredRelations() {
    // 清理过期的关系
    size_t count = 0;
    
    std::unique_ptr<rocksdb::Iterator> it(ltm_db_>NewIterator(rocksdb::ReadOptions(), ltm_cf_relations_));
    for (it->SeekToFirst(); it->Valid(); it->Next()) {
        // 解析关系数据，检查是否需要删除
        // 这里简化处理，删除所有关系
        // 实际应该基于时间衰减和重要性
        ltm_db_>Delete(rocksdb::WriteOptions(), ltm_cf_relations_, it->key());
        count++;
    }
    
    return count;
}

// =============================================================================
// 序列化
// =============================================================================

std::string QuadLayerMemoryStore::serializeMemory(const MemoryTrace& memory) const {
    folly::dynamic obj = folly::dynamic::object;
    
    obj["id"] = memory.id;
    obj["content"] = memory.content;
    obj["type"] = memory.type;
    obj["source"] = memory.source;
    obj["timestamp"] = memory.timestamp;
    obj["created_at"] = memory.created_at;
    obj["updated_at"] = memory.updated_at;
    
    if (memory.embedding.has_value()) {
        folly::dynamic emb = folly::dynamic::array;
        for (float v : memory.embedding.value()) {
            emb.push_back(v);
        }
        obj["embedding"] = emb;
    }
    
    if (memory.location.has_value()) {
        obj["location"] = folly::dynamic::object
            ("latitude", memory.location->latitude)
            ("longitude", memory.location->longitude);
    }
    
    if (!memory.tags.empty()) {
        folly::dynamic tags = folly::dynamic::array;
        for (const auto& tag : memory.tags) {
            tags.push_back(tag);
        }
        obj["tags"] = tags;
    }
    
    return folly::toJson(obj);
}

Result<MemoryTrace> QuadLayerMemoryStore::deserializeMemory(const std::string& data) const {
    try {
        auto obj = folly::parseJson(data);
        
        MemoryTrace memory;
        memory.id = obj["id"].asString();
        memory.content = obj["content"].asString();
        memory.type = obj["type"].asString();
        memory.source = obj["source"].asString();
        memory.timestamp = obj["timestamp"].asInt64();
        memory.created_at = obj["created_at"].asInt64();
        memory.updated_at = obj["updated_at"].asInt64();
        
        if (obj.count("embedding")) {
            Embedding emb;
            for (const auto& v : obj["embedding"]) {
                emb.push_back(static_cast<float>(v.asDouble()));
            }
            memory.embedding = emb;
        }
        
        if (obj.count("location")) {
            GeoPoint loc;
            loc.latitude = obj["location"]["latitude"].asDouble();
            loc.longitude = obj["location"]["longitude"].asDouble();
            memory.location = loc;
        }
        
        if (obj.count("tags")) {
            for (const auto& tag : obj["tags"]) {
                memory.tags.push_back(tag.asString());
            }
        }
        
        return Result<MemoryTrace>(memory);
        
    } catch (const std::exception& e) {
        return Result<MemoryTrace>(ErrorCode::INTERNAL_ERROR,
            std::format("Failed to deserialize memory: {}", e.what()));
    }
}

std::string QuadLayerMemoryStore::serializeStats(const AccessStatistics& stats) const {
    folly::dynamic obj = folly::dynamic::object
        ("access_count", stats.access_count)
        ("last_access", stats.last_access)
        ("first_access", stats.first_access)
        ("importance_score", stats.importance_score);
    
    folly::dynamic history = folly::dynamic::array;
    for (Timestamp t : stats.access_history) {
        history.push_back(t);
    }
    obj["access_history"] = history;
    
    return folly::toJson(obj);
}

Result<AccessStatistics> QuadLayerMemoryStore::deserializeStats(const std::string& data) const {
    try {
        auto obj = folly::parseJson(data);
        
        AccessStatistics stats;
        stats.access_count = obj["access_count"].asInt();
        stats.last_access = obj["last_access"].asInt64();
        stats.first_access = obj["first_access"].asInt64();
        stats.importance_score = obj["importance_score"].asDouble();
        
        if (obj.count("access_history")) {
            for (const auto& t : obj["access_history"]) {
                stats.access_history.push_back(t.asInt64());
            }
        }
        
        return Result<AccessStatistics>(stats);
        
    } catch (const std::exception& e) {
        return Result<AccessStatistics>(ErrorCode::INTERNAL_ERROR,
            std::format("Failed to deserialize stats: {}", e.what()));
    }
}

// =============================================================================
// 缓存管理
// =============================================================================

void QuadLayerMemoryStore::addToLtmCache(const MemoryId& id, const LongTermMemoryEntry& entry) {
    if (ltm_cache_.size() >= config_.long_term.lru_cache_size) {
        // 移除最久未使用的
        auto oldest = ltm_lru_list_.back();
        ltm_lru_list_.pop_back();
        ltm_cache_.erase(oldest);
    }
    
    ltm_cache_[id] = entry;
    ltm_lru_list_.push_front(id);
}

void QuadLayerMemoryStore::removeFromLtmCache(const MemoryId& id) {
    ltm_cache_.erase(id);
    ltm_lru_list_.remove(id);
}

std::optional<LongTermMemoryEntry> QuadLayerMemoryStore::getFromLtmCache(const MemoryId& id) {
    auto it = ltm_cache_.find(id);
    if (it != ltm_cache_.end()) {
        // 更新LRU
        ltm_lru_list_.remove(id);
        ltm_lru_list_.push_front(id);
        return it->second;
    }
    return std::nullopt;
}

// =============================================================================
// 统计和检查点
// =============================================================================

QuadLayerMemoryStore::LayerStatistics QuadLayerMemoryStore::getStatistics() const {
    LayerStatistics stats;
    
    // 感知记忆
    {
        std::shared_lock<std::shared_mutex> lock(sensory_mutex_);
        stats.sensory_count = sensory_buffer_.size();
    }
    
    // 工作记忆
    {
        std::shared_lock<std::shared_mutex> lock(working_mutex_);
        stats.working_count = working_memory_.size();
        stats.working_chunks = chunks_.size();
    }
    
    // 长期记忆
    {
        std::shared_lock<std::shared_mutex> lock(ltm_mutex_);
        // 估算数量
        stats.long_term_count = ltm_cache_.size();
        stats.vector_index_count = 0;  // 需要从HNSW获取
        stats.spatial_index_count = 0;  // 需要从RTree获取
    }
    
    // 参数记忆
    {
        std::shared_lock<std::shared_mutex> lock(param_mutex_);
        stats.parameter_count = parameters_.size();
    }
    
    // 计算命中率
    size_t total_accesses = cache_hits_.load() + cache_misses_.load();
    stats.cache_hit_rate = total_accesses > 0 
        ? static_cast<double>(cache_hits_.load()) / total_accesses 
        : 0.0;
    
    // 平均访问延迟
    stats.avg_access_latency_ms = access_count_.load() > 0 
        ? total_access_time_ms_.load() / access_count_.load() 
        : 0.0;
    
    return stats;
}

Result<bool> QuadLayerMemoryStore::createCheckpoint(const std::string& path) {
    try {
        std::filesystem::create_directories(path);
        
        // 1. 保存HNSW索引
        if (hnsw_index_) {
            hnsw_index_>saveIndex(path + "/hnsw.index");
        }
        
        // 2. 创建RocksDB检查点
        rocksdb::Checkpoint* checkpoint = nullptr;
        auto status = rocksdb::Checkpoint::Create(ltm_db_.get(), &checkpoint);
        if (!status.ok()) {
            return Result<bool>(ErrorCode::STORAGE_WRITE_ERROR,
                std::format("Failed to create checkpoint: {}", status.ToString()));
        }
        
        status = checkpoint->CreateCheckpoint(path + "/ltm_checkpoint");
        delete checkpoint;
        
        if (!status.ok()) {
            return Result<bool>(ErrorCode::STORAGE_WRITE_ERROR,
                std::format("Failed to save checkpoint: {}", status.ToString()));
        }
        
        // 3. 保存配置和元数据
        folly::dynamic metadata = folly::dynamic::object
            ("timestamp", MemoryTrace::now())
            ("version", "2.0")
            ("layers", folly::dynamic::object
                ("sensory", static_cast<int>(stats.sensory_count))
                ("working", static_cast<int>(stats.working_count))
                ("long_term", static_cast<int>(stats.long_term_count))
                ("parameter", static_cast<int>(stats.parameter_count)));
        
        std::ofstream meta_file(path + "/metadata.json");
        meta_file << folly::toJson(metadata);
        meta_file.close();
        
        return Result<bool>(true);
        
    } catch (const std::exception& e) {
        return Result<bool>(ErrorCode::INTERNAL_ERROR,
            std::format("Checkpoint creation failed: {}", e.what()));
    }
}

Result<bool> QuadLayerMemoryStore::restoreFromCheckpoint(const std::string& path) {
    // 这里简化处理，实际需要更复杂的恢复逻辑
    // 包括关闭现有数据库、恢复文件、重新初始化等
    return Result<bool>(ErrorCode::INTERNAL_ERROR, "Restore not fully implemented");
}

} // namespace memory
} // namespace personal_ontology
