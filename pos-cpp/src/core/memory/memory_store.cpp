#include "memory_store.h"
#include <nlohmann/json.hpp>
#include <rocksdb/db.h>
#include <hnswlib/hnswlib.h>
#include <spatialindex/SpatialIndex.h>  // 或使用简单RTree实现
#include <algorithm>
#include <math>
#include <fstream>
#include <iostream>

namespace pos {

using json = nlohmann::json;

// 简单的RTree实现用于空间索引
class SimpleRTree {
public:
    struct Node {
        GeoBoundingBox bbox;
        std::vector<MemoryId> memory_ids;
        std::vector<std::unique_ptr<Node>> children;
        bool is_leaf{true};
    };
    
    std::unique_ptr<Node> root;
    static constexpr size_t MAX_ENTRIES = 8;
    
    SimpleRTree() : root(std::make_unique<Node>()) {}
    
    void insert(const GeoPoint& p, const MemoryId& id) {
        // 简化实现：直接存储到根节点
        root->memory_ids.push_back(id);
        // 扩展bbox
        if (root->bbox.min.latitude == 0 && root->bbox.min.longitude == 0) {
            root->bbox.min = root->bbox.max = p;
        } else {
            root->bbox.min.latitude = std::min(root->bbox.min.latitude, p.latitude);
            root->bbox.min.longitude = std::min(root->bbox.min.longitude, p.longitude);
            root->bbox.max.latitude = std::max(root->bbox.max.latitude, p.latitude);
            root->bbox.max.longitude = std::max(root->bbox.max.longitude, p.longitude);
        }
    }
    
    std::vector<MemoryId> queryRange(const GeoPoint& center, double radius_meters) const {
        std::vector<MemoryId> results;
        // 简化的距离计算
        for (const auto& id : root->memory_ids) {
            results.push_back(id);
        }
        return results;
    }
};

// 实现类
class HierarchicalMemoryStore::Impl {
public:
    std::string data_dir_;
    size_t vector_dim_;
    
    // 存储
    std::unique_ptr<rocksdb::DB> db_;
    
    // 向量索引 (HNSW)
    std::unique_ptr<hnswlib::HierarchicalNSW<float>> hnsw_index_;
    std::unique_ptr<hnswlib::L2Space> hnsw_space_;
    std::unordered_map<MemoryId, size_t> memory_to_hnsw_id_;
    std::unordered_map<size_t, MemoryId> hnsw_id_to_memory_;
    size_t next_hnsw_id_{0};
    
    // 空间索引
    SimpleRTree rtree_;
    
    // 时间索引 (简化：按时间戳排序的列表)
    std::vector<std::pair<Timestamp, MemoryId>> time_index_;
    
    // 本体绑定索引
    std::unordered_map<ConceptId, std::vector<MemoryId>> concept_index_;
    
    Impl(const std::string& dir, size_t dim) : data_dir_(dir), vector_dim_(dim) {
        // 初始化RocksDB
        rocksdb::Options options;
        options.create_if_missing = true;
        options.IncreaseParallelism();
        
        rocksdb::DB* db;
        rocksdb::Status status = rocksdb::DB::Open(options, dir + "/memories", &db);
        if (!status.ok()) {
            throw std::runtime_error("Failed to open memory database: " + status.ToString());
        }
        db_.reset(db);
        
        // 初始化HNSW
        hnsw_space_ = std::make_unique<hnswlib::L2Space>(dim);
        hnsw_index_ = std::make_unique<hnswlib::HierarchicalNSW<float>>(
            hnsw_space_.get(), 10000, 16, 200, 100, false, false
        );
        
        // 加载现有数据
        loadExistingData();
    }
    
    void loadExistingData() {
        rocksdb::ReadOptions read_options;
        std::unique_ptr<rocksdb::Iterator> it(db_->NewIterator(read_options));
        
        for (it->SeekToFirst(); it->Valid(); it->Next()) {
            std::string key = it->key().ToString();
            if (key.substr(0, 2) == "m:") {
                auto memory = MemoryTrace::fromJson(it->value().ToString());
                
                // 重建向量索引
                if (!memory.content.embedding.empty()) {
                    addToVectorIndex(memory);
                }
                
                // 重建空间索引
                if (memory.location) {
                    rtree_.insert(*memory.location, memory.id);
                }
                
                // 重建时间索引
                time_index_.push_back({memory.timestamp, memory.id});
                
                // 重建本体索引
                for (const auto& concept : memory.ontology_bindings) {
                    concept_index_[concept].push_back(memory.id);
                }
            }
        }
        
        // 排序时间索引
        std::sort(time_index_.begin(), time_index_.end());
    }
    
    void addToVectorIndex(const MemoryTrace& memory) {
        if (memory.content.embedding.empty()) return;
        
        size_t hnsw_id = next_hnsw_id_++;
        hnsw_index_->addPoint(memory.content.embedding.data(), hnsw_id, false);
        memory_to_hnsw_id_[memory.id] = hnsw_id;
        hnsw_id_to_memory_[hnsw_id] = memory.id;
    }
    
    std::string memoryKey(const MemoryId& id) const {
        return "m:" + id;
    }
    
    std::string typeKey(MemoryType type) const {
        return "idx:type:" + std::to_string(static_cast<int>(type));
    }
};

// MemoryTrace方法
void MemoryTrace::bindToConcept(const ConceptId& concept) {
    ontology_bindings.insert(concept);
    updated_at = std::chrono::system_clock::now();
}

void MemoryTrace::addAssociation(const MemoryAssociation& assoc) {
    associations.push_back(assoc);
}

void MemoryTrace::updateAccess() {
    access_count++;
    last_accessed = std::chrono::system_clock::now();
}

std::string MemoryTrace::toJson() const {
    json j;
    j["id"] = id;
    j["type"] = static_cast<int>(type);
    j["status"] = static_cast<int>(status);
    j["timestamp"] = timestampToString(timestamp);
    if (end_time) j["end_time"] = timestampToString(*end_time);
    j["temporal_context"] = temporal_context;
    
    if (location) {
        j["location"] = {{"lat", location->latitude}, {"lng", location->longitude}};
    }
    if (location_name) j["location_name"] = *location_name;
    if (spatial_context) j["spatial_context"] = *spatial_context;
    
    j["content"]["raw_text"] = content.raw_text;
    j["content"]["metadata"] = content.metadata;
    // embedding单独存储或不存储在JSON中
    
    j["emotions"] = json::array();
    for (const auto& e : emotions) {
        json ej;
        ej["emotion"] = e.emotion;
        ej["valence"] = e.valence;
        ej["arousal"] = e.arousal;
        j["emotions"].push_back(ej);
    }
    j["overall_valence"] = overall_valence;
    
    j["ontology_bindings"] = ontology_bindings;
    
    j["associations"] = json::array();
    for (const auto& a : associations) {
        json aj;
        aj["target_id"] = a.target_id;
        aj["type"] = a.type;
        aj["strength"] = a.strength;
        aj["description"] = a.description;
        j["associations"].push_back(aj);
    }
    
    j["access_count"] = access_count;
    j["last_accessed"] = timestampToString(last_accessed);
    j["created_at"] = timestampToString(created_at);
    j["importance_score"] = importance_score;
    j["source"] = source;
    j["confidence"] = confidence;
    
    return j.dump(2);
}

MemoryTrace MemoryTrace::fromJson(const std::string& json_str) {
    json j = json::parse(json_str);
    MemoryTrace m;
    
    m.id = j["id"];
    m.type = static_cast<MemoryType>(j.value("type", 0));
    m.status = static_cast<ConsolidationStatus>(j.value("status", 0));
    m.timestamp = parseTimestamp(j["timestamp"]);
    if (j.contains("end_time")) m.end_time = parseTimestamp(j["end_time"]);
    if (j.contains("temporal_context")) m.temporal_context = j["temporal_context"];
    
    if (j.contains("location")) {
        m.location = GeoPoint{
            j["location"]["lat"],
            j["location"]["lng"]
        };
    }
    if (j.contains("location_name")) m.location_name = j["location_name"];
    if (j.contains("spatial_context")) m.spatial_context = j["spatial_context"];
    
    m.content.raw_text = j["content"]["raw_text"];
    if (j["content"].contains("metadata")) {
        m.content.metadata = j["content"]["metadata"];
    }
    
    if (j.contains("emotions")) {
        for (const auto& ej : j["emotions"]) {
            EmotionalTag e;
            e.emotion = ej["emotion"];
            e.valence = ej["valence"];
            e.arousal = ej["arousal"];
            m.emotions.push_back(e);
        }
    }
    m.overall_valence = j.value("overall_valence", 0.0f);
    
    if (j.contains("ontology_bindings")) {
        for (const auto& bid : j["ontology_bindings"]) {
            m.ontology_bindings.insert(bid);
        }
    }
    
    if (j.contains("associations")) {
        for (const auto& aj : j["associations"]) {
            MemoryAssociation a;
            a.target_id = aj["target_id"];
            a.type = aj["type"];
            a.strength = aj["strength"];
            a.description = aj.value("description", "");
            m.associations.push_back(a);
        }
    }
    
    m.access_count = j.value("access_count", 0);
    m.last_accessed = parseTimestamp(j.value("last_accessed", j["timestamp"]));
    m.created_at = parseTimestamp(j["created_at"]);
    m.importance_score = j.value("importance_score", 0.5f);
    m.source = j.value("source", "imported");
    m.confidence = j.value("confidence", 1.0f);
    
    return m;
}

// HierarchicalMemoryStore实现
HierarchicalMemoryStore::HierarchicalMemoryStore(const std::string& data_dir,
                                                  size_t vector_dim)
    : pimpl_(std::make_unique<Impl>(data_dir, vector_dim)) {}

HierarchicalMemoryStore::~HierarchicalMemoryStore() = default;

bool HierarchicalMemoryStore::store(const MemoryTrace& memory) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    
    // 存储到RocksDB
    std::string key = pimpl_->memoryKey(memory.id);
    std::string value = memory.toJson();
    
    rocksdb::Status status = pimpl_->db_->Put(rocksdb::WriteOptions(), key, value);
    if (!status.ok()) {
        return false;
    }
    
    // 添加到向量索引
    if (!memory.content.embedding.empty()) {
        pimpl_->addToVectorIndex(memory);
    }
    
    // 添加到空间索引
    if (memory.location) {
        pimpl_->rtree_.insert(*memory.location, memory.id);
    }
    
    // 添加到时间索引
    pimpl_->time_index_.push_back({memory.timestamp, memory.id});
    std::sort(pimpl_->time_index_.begin(), pimpl_->time_index_.end());
    
    // 更新本体索引
    for (const auto& concept : memory.ontology_bindings) {
        pimpl_->concept_index_[concept].push_back(memory.id);
    }
    
    return true;
}

bool HierarchicalMemoryStore::storeBatch(const std::vector<MemoryTrace>& memories) {
    for (const auto& m : memories) {
        if (!store(m)) return false;
    }
    return true;
}

std::optional<MemoryTrace> HierarchicalMemoryStore::getById(const MemoryId& id) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    
    std::string value;
    rocksdb::Status status = pimpl_->db_->Get(
        rocksdb::ReadOptions(),
        pimpl_->memoryKey(id),
        &value
    );
    
    if (!status.ok()) {
        return std::nullopt;
    }
    
    return MemoryTrace::fromJson(value);
}

std::vector<MemoryRetrievalResult> HierarchicalMemoryStore::retrieveByText(
    const std::string& query,
    const Embedding& query_embedding,
    int limit) {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    
    std::vector<MemoryRetrievalResult> results;
    
    if (query_embedding.empty() || pimpl_->next_hnsw_id_ == 0) {
        return results;
    }
    
    // HNSW搜索
    std::vector<std::pair<float, size_t>> knn = pimpl_->hnsw_index_->searchKnn(
        query_embedding.data(),
        limit * 2,  // 获取更多以便过滤
        nullptr
    );
    
    for (const auto& [dist, hnsw_id] : knn) {
        auto it = pimpl_->hnsw_id_to_memory_.find(hnsw_id);
        if (it == pimpl_->hnsw_id_to_memory_.end()) continue;
        
        auto memory_opt = getById(it->second);
        if (!memory_opt) continue;
        
        MemoryRetrievalResult result;
        result.memory = *memory_opt;
        result.semantic_score = 1.0f / (1.0f + dist);  // 转换为相似度
        result.relevance_score = result.semantic_score;
        result.match_type = "semantic";
        
        results.push_back(result);
        if (results.size() >= static_cast<size_t>(limit)) break;
    }
    
    return results;
}

std::vector<MemoryRetrievalResult> HierarchicalMemoryStore::retrieveByTime(
    const Timestamp& start,
    const Timestamp& end,
    int limit) {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    
    std::vector<MemoryRetrievalResult> results;
    
    for (const auto& [ts, id] : pimpl_->time_index_) {
        if (ts >= start && ts <= end) {
            auto memory_opt = getById(id);
            if (memory_opt) {
                MemoryRetrievalResult result;
                result.memory = *memory_opt;
                result.temporal_score = 1.0f;
                result.relevance_score = 1.0f;
                result.match_type = "temporal";
                results.push_back(result);
                
                if (results.size() >= static_cast<size_t>(limit)) break;
            }
        }
        if (ts > end) break;  // 时间已排序
    }
    
    return results;
}

std::vector<MemoryRetrievalResult> HierarchicalMemoryStore::retrieveByLocation(
    const GeoPoint& center,
    double radius_meters,
    int limit) {
    std::vector<MemoryRetrievalResult> results;
    
    // 获取候选
    auto candidate_ids = pimpl_->rtree_.queryRange(center, radius_meters);
    
    for (const auto& id : candidate_ids) {
        auto memory_opt = getById(id);
        if (!memory_opt || !memory_opt->location) continue;
        
        double dist = center.distanceTo(*memory_opt->location);
        if (dist <= radius_meters) {
            MemoryRetrievalResult result;
            result.memory = *memory_opt;
            result.spatial_score = 1.0f - static_cast<float>(dist / radius_meters);
            result.relevance_score = result.spatial_score;
            result.match_type = "spatial";
            results.push_back(result);
            
            if (results.size() >= static_cast<size_t>(limit)) break;
        }
    }
    
    return results;
}

std::vector<MemoryRetrievalResult> HierarchicalMemoryStore::retrieveByConcepts(
    const std::vector<ConceptId>& concepts,
    bool require_all,
    int limit) {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    
    std::vector<MemoryRetrievalResult> results;
    std::unordered_map<MemoryId, int> concept_match_count;
    
    // 统计每个记忆匹配的概念数
    for (const auto& concept : concepts) {
        auto it = pimpl_->concept_index_.find(concept);
        if (it != pimpl_->concept_index_.end()) {
            for (const auto& memory_id : it->second) {
                concept_match_count[memory_id]++;
            }
        }
    }
    
    // 筛选
    for (const auto& [memory_id, count] : concept_match_count) {
        bool match = require_all ? 
            (count >= static_cast<int>(concepts.size())) : 
            (count > 0);
        
        if (match) {
            auto memory_opt = getById(memory_id);
            if (memory_opt) {
                MemoryRetrievalResult result;
                result.memory = *memory_opt;
                result.relevance_score = static_cast<float>(count) / concepts.size();
                result.match_type = "conceptual";
                results.push_back(result);
                
                if (results.size() >= static_cast<size_t>(limit)) break;
            }
        }
    }
    
    return results;
}

std::vector<MemoryRetrievalResult> HierarchicalMemoryStore::retrieveContextual(
    const std::optional<Timestamp>& time_hint,
    const std::optional<GeoPoint>& location_hint,
    const std::vector<ConceptId>& concept_hints,
    const std::optional<std::string>& text_hint,
    int limit) {
    
    // 多路召回
    std::vector<MemoryRetrievalResult> candidates;
    
    // 1. 概念检索
    if (!concept_hints.empty()) {
        auto concept_results = retrieveByConcepts(concept_hints, false, limit * 2);
        candidates.insert(candidates.end(), concept_results.begin(), concept_results.end());
    }
    
    // 2. 时间检索
    if (time_hint) {
        auto start = *time_hint - std::chrono::hours(24);
        auto end = *time_hint + std::chrono::hours(24);
        auto time_results = retrieveByTime(start, end, limit);
        candidates.insert(candidates.end(), time_results.begin(), time_results.end());
    }
    
    // 3. 空间检索
    if (location_hint) {
        auto spatial_results = retrieveByLocation(*location_hint, 5000.0, limit);  // 5km
        candidates.insert(candidates.end(), spatial_results.begin(), spatial_results.end());
    }
    
    // 去重并融合
    std::unordered_map<MemoryId, MemoryRetrievalResult> fused;
    for (auto& r : candidates) {
        auto it = fused.find(r.memory.id);
        if (it == fused.end()) {
            fused[r.memory.id] = r;
        } else {
            // 合并分数
            it->second.relevance_score = std::max(it->second.relevance_score, r.relevance_score);
            it->second.semantic_score = std::max(it->second.semantic_score, r.semantic_score);
            it->second.temporal_score = std::max(it->second.temporal_score, r.temporal_score);
            it->second.spatial_score = std::max(it->second.spatial_score, r.spatial_score);
        }
    }
    
    // 转换为向量并排序
    std::vector<MemoryRetrievalResult> results;
    for (auto& [id, r] : fused) {
        // 综合分数计算
        r.relevance_score = (r.semantic_score + r.temporal_score + r.spatial_score) / 3.0f;
        if (r.relevance_score > 0) {
            results.push_back(r);
        }
    }
    
    std::sort(results.begin(), results.end(),
        [](const auto& a, const auto& b) {
            return a.relevance_score > b.relevance_score;
        });
    
    if (results.size() > static_cast<size_t>(limit)) {
        results.resize(limit);
    }
    
    return results;
}

bool HierarchicalMemoryStore::updateAccessStats(const MemoryId& id) {
    auto memory_opt = getById(id);
    if (!memory_opt) return false;
    
    auto memory = *memory_opt;
    memory.updateAccess();
    
    return store(memory);
}

bool HierarchicalMemoryStore::bindToConcept(const MemoryId& memory_id, 
                                           const ConceptId& concept) {
    auto memory_opt = getById(memory_id);
    if (!memory_opt) return false;
    
    auto memory = *memory_opt;
    memory.bindToConcept(concept);
    
    return store(memory);
}

size_t HierarchicalMemoryStore::getMemoryCount() const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    return pimpl_->time_index_.size();
}

bool HierarchicalMemoryStore::forget(const MemoryId& id, bool permanent) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    
    if (permanent) {
        pimpl_->db_->Delete(rocksdb::WriteOptions(), pimpl_->memoryKey(id));
    } else {
        // 软删除：标记为遗忘状态
        auto memory_opt = getById(id);
        if (!memory_opt) return false;
        
        auto memory = *memory_opt;
        memory.status = ConsolidationStatus::FORGOTTEN;
        store(memory);
    }
    
    return true;
}

} // namespace pos
