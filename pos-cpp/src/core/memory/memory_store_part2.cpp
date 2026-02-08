/**
 * @file memory_store_part2.cpp
 * @brief 分层记忆存储系统实现 - 第二部分 (查询和索引)
 */

#include "memory_store.h"
#include <hnswlib/hnswlib.h>
#include <rocksdb/write_batch.h>
#include <folly/json.h>

namespace personal_ontology {
namespace memory {

// =============================================================================
// 查询操作
// =============================================================================

Result<MemoryQueryResult> HierarchicalMemoryStore::query(const MemoryQuery& query) {
    MemoryQueryResult result;
    std::set<MemoryId> unique_ids;  // 用于去重
    
    // 1. 在工作记忆中查询
    {
        std::shared_lock<std::shared_mutex> lock(wm_mutex_);
        for (const auto& [id, entry] : working_memory_) {
            if (matchesQuery(entry.memory, query) && 
                unique_ids.insert(id).second) {
                result.items.push_back(entry.memory);
                if (result.items.size() >= query.limit) {
                    result.total_count = result.items.size();
                    result.has_more = true;
                    return Result<MemoryQueryResult>(std::move(result));
                }
            }
        }
    }
    
    // 2. 在短期记忆中查询
    {
        std::shared_lock<std::shared_mutex> lock(stm_mutex_);
        
        // 先查缓存
        for (const auto& [id, entry] : short_term_cache_) {
            if (matchesQuery(entry.memory, query) && 
                unique_ids.insert(id).second) {
                result.items.push_back(entry.memory);
                if (result.items.size() >= query.limit) {
                    result.total_count = result.items.size();
                    result.has_more = true;
                    return Result<MemoryQueryResult>(std::move(result));
                }
            }
        }
    }
    
    // 3. 在长期记忆中查询 (使用RocksDB迭代器)
    if (ltm_db_ && result.items.size() < query.limit) {
        rocksdb::ReadOptions read_options;
        std::unique_ptr<rocksdb::Iterator> it(ltm_db_>NewIterator(read_options, ltm_cf_memories_));
        
        size_t skipped = 0;
        for (it->SeekToFirst(); it->Valid(); it->Next()) {
            auto memory_result = deserializeMemory(it->value().ToString());
            if (memory_result.isError()) continue;
            
            const auto& memory = memory_result.value();
            if (matchesQuery(memory, query) && 
                unique_ids.insert(memory.id).second) {
                
                if (skipped < query.offset) {
                    skipped++;
                    continue;
                }
                
                result.items.push_back(memory);
                if (result.items.size() >= query.limit) {
                    result.has_more = true;
                    break;
                }
            }
        }
    }
    
    // 排序
    sortResults(result.items, query.sort_by);
    
    result.total_count = result.items.size();
    return Result<MemoryQueryResult>(std::move(result));
}

// 辅助函数：检查记忆是否匹配查询条件
bool matchesQuery(const MemoryTrace& memory, const MemoryQuery& query) {
    // 类型过滤
    if (query.type_filter.has_value() && 
        memory.memory_type != query.type_filter.value()) {
        return false;
    }
    
    // 内容关键词过滤
    if (query.content_keyword.has_value()) {
        const auto& keyword = query.content_keyword.value();
        if (memory.content.find(keyword) == std::string::npos) {
            // 也检查标签
            bool found_in_tags = false;
            for (const auto& tag : memory.tags) {
                if (tag.find(keyword) != std::string::npos) {
                    found_in_tags = true;
                    break;
                }
            }
            if (!found_in_tags) return false;
        }
    }
    
    // 时间范围过滤
    if (query.time_start.has_value()) {
        Timestamp time = memory.event_time.has_value() ? 
                        memory.event_time.value() : memory.created_at;
        if (time < query.time_start.value()) return false;
    }
    if (query.time_end.has_value()) {
        Timestamp time = memory.event_time.has_value() ? 
                        memory.event_time.value() : memory.created_at;
        if (time > query.time_end.value()) return false;
    }
    
    // 空间范围过滤
    if (query.spatial_bounds.has_value() && memory.location.has_value()) {
        if (!query.spatial_bounds.value().contains(memory.location.value())) {
            return false;
        }
    }
    
    // 标签过滤
    if (!query.tags_filter.empty()) {
        for (const auto& tag : query.tags_filter) {
            if (std::find(memory.tags.begin(), memory.tags.end(), tag) == memory.tags.end()) {
                return false;
            }
        }
    }
    
    // 概念过滤
    if (!query.concept_filter.empty()) {
        for (const auto& concept_id : query.concept_filter) {
            if (std::find(memory.related_concepts.begin(), 
                         memory.related_concepts.end(), 
                         concept_id) == memory.related_concepts.end()) {
                return false;
            }
        }
    }
    
    // 置信度过滤
    if (query.min_confidence.has_value() && 
        memory.confidence < query.min_confidence.value()) {
        return false;
    }
    
    return true;
}

// 辅助函数：排序结果
void sortResults(std::vector<MemoryTrace>& memories, MemoryQuery::SortBy sort_by) {
    switch (sort_by) {
        case MemoryQuery::SortBy::TIME_DESC:
            std::sort(memories.begin(), memories.end(), 
                [](const MemoryTrace& a, const MemoryTrace& b) {
                    Timestamp ta = a.event_time.has_value() ? a.event_time.value() : a.created_at;
                    Timestamp tb = b.event_time.has_value() ? b.event_time.value() : b.created_at;
                    return ta > tb;
                });
            break;
            
        case MemoryQuery::SortBy::TIME_ASC:
            std::sort(memories.begin(), memories.end(), 
                [](const MemoryTrace& a, const MemoryTrace& b) {
                    Timestamp ta = a.event_time.has_value() ? a.event_time.value() : a.created_at;
                    Timestamp tb = b.event_time.has_value() ? b.event_time.value() : b.created_at;
                    return ta < tb;
                });
            break;
            
        case MemoryQuery::SortBy::IMPORTANCE:
            std::sort(memories.begin(), memories.end(), 
                [](const MemoryTrace& a, const MemoryTrace& b) {
                    return a.importance_score > b.importance_score;
                });
            break;
            
        case MemoryQuery::SortBy::ACCESS_COUNT:
            std::sort(memories.begin(), memories.end(), 
                [](const MemoryTrace& a, const MemoryTrace& b) {
                    return a.access_count > b.access_count;
                });
            break;
            
        case MemoryQuery::SortBy::RELEVANCE:
            // 相关性排序需要向量查询，这里默认按时间降序
            std::sort(memories.begin(), memories.end(), 
                [](const MemoryTrace& a, const MemoryTrace& b) {
                    return a.created_at > b.created_at;
                });
            break;
    }
}

Result<std::vector<ScoredItem<MemoryTrace>>> HierarchicalMemoryStore::semanticSearch(
    const Embedding& embedding, size_t top_k, double min_score) {
    
    std::vector<ScoredItem<MemoryTrace>> results;
    
    // 1. 使用HNSW索引搜索
    auto hnsw_result = searchHnsw(embedding, top_k * 2);  // 获取更多结果用于过滤
    if (hnsw_result.isError()) {
        return Result<std::vector<ScoredItem<MemoryTrace>>>(
            hnsw_result.errorCode(), hnsw_result.errorMessage());
    }
    
    const auto& candidates = hnsw_result.value();
    
    // 2. 获取记忆并计算相似度
    for (const auto& [id, distance] : candidates) {
        // 转换距离为相似度分数 (假设使用L2距离)
        double score = 1.0 / (1.0 + distance);
        
        if (score >= min_score) {
            auto memory_result = retrieve(id);
            if (memory_result.isOk()) {
                ScoredItem<MemoryTrace> item;
                item.item = memory_result.value();
                item.score = score;
                results.push_back(item);
                
                if (results.size() >= top_k) break;
            }
        }
    }
    
    // 3. 按分数排序
    std::sort(results.begin(), results.end(), 
        [](const ScoredItem<MemoryTrace>& a, const ScoredItem<MemoryTrace>& b) {
            return a.score > b.score;
        });
    
    return Result<std::vector<ScoredItem<MemoryTrace>>>(std::move(results));
}

Result<std::vector<ScoredItem<MemoryTrace>>> HierarchicalMemoryStore::multimodalSearch(
    const Embedding& embedding, const MemoryQuery& query, size_t top_k) {
    
    // 先进行语义搜索
    auto search_result = semanticSearch(embedding, top_k * 3, 0.0);
    if (search_result.isError()) {
        return search_result;
    }
    
    auto candidates = search_result.value();
    std::vector<ScoredItem<MemoryTrace>> filtered;
    
    // 应用元数据过滤
    for (auto& candidate : candidates) {
        if (matchesQuery(candidate.item, query)) {
            filtered.push_back(std::move(candidate));
            if (filtered.size() >= top_k) break;
        }
    }
    
    return Result<std::vector<ScoredItem<MemoryTrace>>>(std::move(filtered));
}

Result<std::vector<MemoryTrace>> HierarchicalMemoryStore::spatiotemporalQuery(
    std::optional<Timestamp> time_start,
    std::optional<Timestamp> time_end,
    std::optional<GeoBoundingBox> bbox,
    size_t limit) {
    
    std::vector<MemoryTrace> results;
    std::set<MemoryId> candidate_ids;
    
    // 1. 空间查询
    if (bbox.has_value()) {
        auto spatial_results = searchSpatial(bbox.value());
        for (const auto& id : spatial_results) {
            candidate_ids.insert(id);
        }
    }
    
    // 2. 时间查询
    if (time_start.has_value() || time_end.has_value()) {
        Timestamp start = time_start.value_or(0);
        Timestamp end = time_end.value_or(std::numeric_limits<Timestamp>::max());
        auto temporal_results = searchTemporal(start, end);
        
        if (candidate_ids.empty()) {
            // 只有时间查询
            for (const auto& id : temporal_results) {
                candidate_ids.insert(id);
            }
        } else {
            // 时空联合查询，取交集
            std::set<MemoryId> temporal_set(temporal_results.begin(), temporal_results.end());
            std::set<MemoryId> intersection;
            std::set_intersection(candidate_ids.begin(), candidate_ids.end(),
                                temporal_set.begin(), temporal_set.end(),
                                std::inserter(intersection, intersection.begin()));
            candidate_ids = std::move(intersection);
        }
    }
    
    // 3. 获取记忆
    for (const auto& id : candidate_ids) {
        auto result = retrieve(id);
        if (result.isOk()) {
            results.push_back(result.value());
            if (results.size() >= limit) break;
        }
    }
    
    return Result<std::vector<MemoryTrace>>(std::move(results));
}

Result<std::vector<MemoryTrace>> HierarchicalMemoryStore::getEntityMemories(
    const EntityId& entity_id, size_t limit) {
    
    std::vector<MemoryTrace> results;
    MemoryQuery query;
    query.limit = limit;
    
    // 扫描所有存储层查找匹配的记忆
    auto query_result = query(query);
    if (query_result.isError()) {
        return Result<std::vector<MemoryTrace>>(
            query_result.errorCode(), query_result.errorMessage());
    }
    
    for (auto& memory : query_result.value().items) {
        if (memory.entity_id == entity_id) {
            results.push_back(std::move(memory));
            if (results.size() >= limit) break;
        }
    }
    
    return Result<std::vector<MemoryTrace>>(std::move(results));
}

Result<std::vector<MemoryTrace>> HierarchicalMemoryStore::getConceptMemories(
    const ConceptId& concept_id, size_t limit) {
    
    MemoryQuery query;
    query.concept_filter = {concept_id};
    query.limit = limit;
    
    return query(query).transform([](MemoryQueryResult& result) {
        return std::move(result.items);
    });
}

// =============================================================================
// 记忆管理
// =============================================================================

Result<size_t> HierarchicalMemoryStore::consolidate() {
    size_t consolidated_count = 0;
    
    // 1. 从短期记忆中选择重要记忆迁移到长期记忆
    {
        std::unique_lock<std::shared_mutex> lock(stm_mutex_);
        
        std::vector<MemoryId> to_migrate;
        for (const auto& [id, entry] : short_term_cache_) {
            // 选择重要性高或访问频繁的记忆
            if (entry.stats.importance_score > 0.7 || entry.stats.access_count > 5) {
                to_migrate.push_back(id);
            }
        }
        
        // 迁移到长期记忆
        for (const auto& id : to_migrate) {
            auto result = migrateToLongTerm(id);
            if (result.isOk()) {
                consolidated_count++;
            }
        }
    }
    
    // 2. 从工作记忆迁移较旧的记忆到短期记忆
    {
        std::unique_lock<std::shared_mutex> lock(wm_mutex_);
        
        auto now = MemoryTrace::now();
        std::vector<MemoryId> to_demote;
        
        for (const auto& [id, entry] : working_memory_) {
            if (now - entry.memory.created_at > config_.working_memory_max_age_ms) {
                to_demote.push_back(id);
            }
        }
        
        for (const auto& id : to_demote) {
            migrateToShortTerm(id);
        }
    }
    
    return Result<size_t>(consolidated_count);
}

Result<size_t> HierarchicalMemoryStore::forget() {
    size_t forgotten_count = 0;
    
    // 1. 从工作记忆中移除低重要性记忆
    {
        std::unique_lock<std::shared_mutex> lock(wm_mutex_);
        
        std::vector<MemoryId> to_forget;
        for (const auto& [id, entry] : working_memory_) {
            if (entry.stats.importance_score < 0.2 && entry.stats.access_count < 2) {
                to_forget.push_back(id);
            }
        }
        
        for (const auto& id : to_forget) {
            remove(id);
            forgotten_count++;
        }
    }
    
    // 2. 从短期记忆中清除不重要的记忆
    {
        std::unique_lock<std::shared_mutex> lock(stm_mutex_);
        
        rocksdb::ReadOptions read_options;
        std::unique_ptr<rocksdb::Iterator> it(stm_db_>NewIterator(read_options, stm_cf_default_));
        rocksdb::WriteBatch batch;
        
        for (it->SeekToFirst(); it->Valid(); it->Next()) {
            auto stats_result = deserializeAccessStats(
                it->value().ToString());
            if (stats_result.isOk()) {
                const auto& stats = stats_result.value();
                if (stats.importance_score < 0.15 && stats.access_count < 1) {
                    batch.Delete(stm_cf_default_, it->key());
                    batch.Delete(stm_cf_metadata_, it->key().ToString() + ":stats");
                    forgotten_count++;
                }
            }
        }
        
        rocksdb::WriteOptions write_options;
        stm_db_>Write(write_options, &batch);
    }
    
    return Result<size_t>(forgotten_count);
}

Result<size_t> HierarchicalMemoryStore::cleanupExpired() {
    size_t cleaned_count = 0;
    
    // 检查所有存储层中的过期记忆
    auto remove_if_expired = [&](const MemoryId& id, const MemoryTrace& memory) {
        if (memory.isExpired()) {
            remove(id);
            return true;
        }
        return false;
    };
    
    // 工作记忆
    {
        std::shared_lock<std::shared_mutex> lock(wm_mutex_);
        for (const auto& [id, entry] : working_memory_) {
            if (remove_if_expired(id, entry.memory)) {
                cleaned_count++;
            }
        }
    }
    
    // 短期记忆
    if (stm_db_) {
        rocksdb::ReadOptions read_options;
        std::unique_ptr<rocksdb::Iterator> it(stm_db_>NewIterator(read_options, stm_cf_default_));
        rocksdb::WriteBatch batch;
        
        for (it->SeekToFirst(); it->Valid(); it->Next()) {
            auto memory_result = deserializeMemory(it->value().ToString());
            if (memory_result.isOk() && memory_result.value().isExpired()) {
                batch.Delete(stm_cf_default_, it->key());
                batch.Delete(stm_cf_metadata_, it->key().ToString() + ":stats");
                cleaned_count++;
            }
        }
        
        rocksdb::WriteOptions write_options;
        stm_db_>Write(write_options, &batch);
    }
    
    // 长期记忆
    if (ltm_db_) {
        rocksdb::ReadOptions read_options;
        std::unique_ptr<rocksdb::Iterator> it(ltm_db_>NewIterator(read_options, ltm_cf_memories_));
        rocksdb::WriteBatch batch;
        
        for (it->SeekToFirst(); it->Valid(); it->Next()) {
            auto memory_result = deserializeMemory(it->value().ToString());
            if (memory_result.isOk() && memory_result.value().isExpired()) {
                batch.Delete(ltm_cf_memories_, it->key());
                batch.Delete(ltm_cf_embeddings_, it->key());
                cleaned_count++;
            }
        }
        
        rocksdb::WriteOptions write_options;
        ltm_db_>Write(write_options, &batch);
    }
    
    return Result<size_t>(cleaned_count);
}

Result<bool> HierarchicalMemoryStore::boostImportance(
    const MemoryId& id, double boost_factor) {
    
    return update(id, [boost_factor](MemoryTrace& memory) {
        memory.importance_score = std::min(1.0f, 
            static_cast<float>(memory.importance_score * boost_factor));
    });
}

Result<bool> HierarchicalMemoryStore::associate(
    const MemoryId& id1, const MemoryId& id2, double strength) {
    
    // 更新两个记忆的相关记忆列表
    auto result1 = update(id1, [id2, strength](MemoryTrace& memory) {
        if (std::find(memory.related_memories.begin(), 
                     memory.related_memories.end(), id2) == memory.related_memories.end()) {
            memory.related_memories.push_back(id2);
        }
    });
    
    auto result2 = update(id2, [id1, strength](MemoryTrace& memory) {
        if (std::find(memory.related_memories.begin(), 
                     memory.related_memories.end(), id1) == memory.related_memories.end()) {
            memory.related_memories.push_back(id1);
        }
    });
    
    return Result<bool>(result1.isOk() && result2.isOk());
}
