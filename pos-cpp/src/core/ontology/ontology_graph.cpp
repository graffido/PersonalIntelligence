#include "ontology_graph.h"
#include <nlohmann/json.hpp>
#include <rocksdb/db.h>
#include <rocksdb/write_batch.h>
#include <queue>
#include <set>
#include <fstream>
#include <iostream>

namespace pos {

using json = nlohmann::json;

// 内部实现类
class OntologyGraph::Impl {
public:
    std::unique_ptr<rocksdb::DB> db_;
    std::string db_path_;
    
    explicit Impl(const std::string& path) : db_path_(path) {
        rocksdb::Options options;
        options.create_if_missing = true;
        options.IncreaseParallelism();
        options.OptimizeLevelStyleCompaction();
        
        rocksdb::DB* db;
        rocksdb::Status status = rocksdb::DB::Open(options, path, &db);
        if (!status.ok()) {
            throw std::runtime_error("Failed to open ontology database: " + 
                                   status.ToString());
        }
        db_.reset(db);
    }
    
    ~Impl() = default;
    
    // 键生成
    std::string conceptKey(const ConceptId& id) const {
        return "c:" + id;
    }
    
    std::string labelIndexKey(const std::string& label) const {
        return "idx:label:" + label;
    }
    
    std::string typeIndexKey(ConceptType type) const {
        return "idx:type:" + conceptTypeToString(type);
    }
    
    std::string relationKey(const ConceptId& from, const ConceptId& to) const {
        return "r:" + from + ":" + to;
    }
    
    std::string metaKey(const std::string& key) const {
        return "meta:" + key;
    }
};

// OntologyConcept方法实现
void OntologyConcept::bindMemory(const MemoryId& memory_id) {
    bound_memories.insert(memory_id);
    updated_at = std::chrono::system_clock::now();
}

void OntologyConcept::unbindMemory(const MemoryId& memory_id) {
    bound_memories.erase(memory_id);
    updated_at = std::chrono::system_clock::now();
}

void OntologyConcept::addRelation(const Relation& rel) {
    relations.push_back(rel);
    updated_at = std::chrono::system_clock::now();
}

bool OntologyConcept::hasRelationTo(const ConceptId& target, RelationType type) const {
    for (const auto& rel : relations) {
        if (rel.target == target && rel.type == type) {
            return true;
        }
    }
    return false;
}

std::string OntologyConcept::toJson() const {
    json j;
    j["id"] = id;
    j["type"] = conceptTypeToString(type);
    j["label"] = label;
    j["description"] = description;
    
    // 属性
    for (const auto& [key, val] : properties) {
        std::visit([&j, &key](auto&& arg) {
            j["properties"][key] = arg;
        }, val);
    }
    
    // 关系
    j["relations"] = json::array();
    for (const auto& rel : relations) {
        json rj;
        rj["type"] = relationTypeToString(rel.type);
        rj["target"] = rel.target;
        rj["weight"] = rel.weight;
        if (rel.temporal_constraint) {
            rj["temporal_constraint"] = *rel.temporal_constraint;
        }
        j["relations"].push_back(rj);
    }
    
    // 记忆绑定
    j["bound_memories"] = bound_memories;
    
    // 元数据
    j["created_at"] = timestampToString(created_at);
    j["updated_at"] = timestampToString(updated_at);
    j["confidence"] = confidence;
    j["source"] = source;
    j["access_count"] = access_count;
    
    return j.dump(2);
}

OntologyConcept OntologyConcept::fromJson(const std::string& json_str) {
    json j = json::parse(json_str);
    OntologyConcept concept;
    
    concept.id = j["id"];
    concept.type = stringToConceptType(j["type"]);
    concept.label = j["label"];
    concept.description = j.value("description", "");
    
    // 解析属性
    if (j.contains("properties")) {
        for (auto& [key, val] : j["properties"].items()) {
            if (val.is_string()) {
                concept.properties[key] = val.get<std::string>();
            } else if (val.is_number_integer()) {
                concept.properties[key] = val.get<int>();
            } else if (val.is_number_float()) {
                concept.properties[key] = val.get<double>();
            } else if (val.is_boolean()) {
                concept.properties[key] = val.get<bool>();
            }
        }
    }
    
    // 解析关系
    if (j.contains("relations")) {
        for (const auto& rj : j["relations"]) {
            Relation rel;
            rel.type = stringToRelationType(rj["type"]);
            rel.target = rj["target"];
            rel.weight = rj.value("weight", 1.0f);
            if (rj.contains("temporal_constraint")) {
                rel.temporal_constraint = rj["temporal_constraint"];
            }
            concept.relations.push_back(rel);
        }
    }
    
    // 解析记忆绑定
    if (j.contains("bound_memories")) {
        for (const auto& m : j["bound_memories"]) {
            concept.bound_memories.insert(m);
        }
    }
    
    concept.created_at = parseTimestamp(j["created_at"]);
    concept.updated_at = parseTimestamp(j["updated_at"]);
    concept.confidence = j.value("confidence", 1.0f);
    concept.source = j.value("source", "imported");
    concept.access_count = j.value("access_count", 0);
    
    return concept;
}

// OntologyGraph实现
OntologyGraph::OntologyGraph(const std::string& storage_path)
    : pimpl_(std::make_unique<Impl>(storage_path)) {}

OntologyGraph::~OntologyGraph() = default;

ConceptId OntologyGraph::createConcept(const std::string& label, ConceptType type,
                                       const std::string& source) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    ConceptId id = generateUUID();
    OntologyConcept concept;
    concept.id = id;
    concept.type = type;
    concept.label = label;
    concept.created_at = std::chrono::system_clock::now();
    concept.updated_at = concept.created_at;
    concept.source = source;
    
    // 存储概念
    std::string key = pimpl_->conceptKey(id);
    std::string value = concept.toJson();
    pimpl_->db_->Put(rocksdb::WriteOptions(), key, value);
    
    // 更新标签索引
    std::string labelKey = pimpl_->labelIndexKey(label);
    std::string existing;
    pimpl_->db_->Get(rocksdb::ReadOptions(), labelKey, &existing);
    if (!existing.empty()) {
        existing += "," + id;
    } else {
        existing = id;
    }
    pimpl_->db_->Put(rocksdb::WriteOptions(), labelKey, existing);
    
    // 更新类型索引
    std::string typeKey = pimpl_->typeIndexKey(type);
    pimpl_->db_->Get(rocksdb::ReadOptions(), typeKey, &existing);
    if (!existing.empty()) {
        existing += "," + id;
    } else {
        existing = id;
    }
    pimpl_->db_->Put(rocksdb::WriteOptions(), typeKey, existing);
    
    // 更新计数
    std::string countStr;
    pimpl_->db_->Get(rocksdb::ReadOptions(), pimpl_->metaKey("concept_count"), &countStr);
    int count = countStr.empty() ? 0 : std::stoi(countStr);
    pimpl_->db_->Put(rocksdb::WriteOptions(), 
                       pimpl_->metaKey("concept_count"), 
                       std::to_string(count + 1));
    
    return id;
}

std::optional<OntologyConcept> OntologyGraph::getConcept(const ConceptId& id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::string value;
    rocksdb::Status status = pimpl_->db_->Get(rocksdb::ReadOptions(), 
                                               pimpl_->conceptKey(id), &value);
    
    if (!status.ok()) {
        return std::nullopt;
    }
    
    auto concept = OntologyConcept::fromJson(value);
    concept.access_count++;
    
    // 异步更新访问计数（简化实现）
    const_cast<OntologyGraph*>(this)->pimpl_->db_->Put(
        rocksdb::WriteOptions(),
        pimpl_->conceptKey(id),
        concept.toJson()
    );
    
    return concept;
}

std::vector<OntologyConcept> OntologyGraph::findConceptsByLabel(
    const std::string& label, int limit) const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::vector<OntologyConcept> results;
    std::string indexValue;
    
    rocksdb::Status status = pimpl_->db_->Get(
        rocksdb::ReadOptions(),
        pimpl_->labelIndexKey(label),
        &indexValue
    );
    
    if (!status.ok() || indexValue.empty()) {
        return results;
    }
    
    // 分割逗号分隔的ID列表
    std::stringstream ss(indexValue);
    std::string id;
    int count = 0;
    while (std::getline(ss, id, ',') && count < limit) {
        auto concept = getConcept(id);
        if (concept) {
            results.push_back(*concept);
            count++;
        }
    }
    
    return results;
}

std::vector<OntologyConcept> OntologyGraph::findConceptsByType(
    ConceptType type, int limit) const {
    std::vector<OntologyConcept> results;
    std::string indexValue;
    
    rocksdb::Status status = pimpl_->db_->Get(
        rocksdb::ReadOptions(),
        pimpl_->typeIndexKey(type),
        &indexValue
    );
    
    if (!status.ok() || indexValue.empty()) {
        return results;
    }
    
    std::stringstream ss(indexValue);
    std::string id;
    int count = 0;
    while (std::getline(ss, id, ',') && count < limit) {
        auto concept = getConcept(id);
        if (concept) {
            results.push_back(*concept);
            count++;
        }
    }
    
    return results;
}

bool OntologyGraph::updateConcept(const OntologyConcept& concept) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto updated = concept;
    updated.updated_at = std::chrono::system_clock::now();
    
    rocksdb::Status status = pimpl_->db_->Put(
        rocksdb::WriteOptions(),
        pimpl_->conceptKey(concept.id),
        updated.toJson()
    );
    
    return status.ok();
}

bool OntologyGraph::deleteConcept(const ConceptId& id) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // 获取概念以清理索引
    auto concept_opt = getConcept(id);
    if (!concept_opt) {
        return false;
    }
    
    auto concept = *concept_opt;
    
    // 删除标签索引
    std::string labelKey = pimpl_->labelIndexKey(concept.label);
    std::string labelIndex;
    pimpl_->db_->Get(rocksdb::ReadOptions(), labelKey, &labelIndex);
    // TODO: 从索引中移除该ID
    
    // 删除类型索引
    std::string typeKey = pimpl_->typeIndexKey(concept.type);
    std::string typeIndex;
    pimpl_->db_->Get(rocksdb::ReadOptions(), typeKey, &typeIndex);
    // TODO: 从索引中移除该ID
    
    // 删除概念
    rocksdb::Status status = pimpl_->db_->Delete(
        rocksdb::WriteOptions(),
        pimpl_->conceptKey(id)
    );
    
    return status.ok();
}

bool OntologyGraph::addRelation(const ConceptId& from, const ConceptId& to,
                               RelationType type, float weight,
                               const std::optional<std::string>& temporal) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto from_opt = getConcept(from);
    if (!from_opt) return false;
    
    auto from_concept = *from_opt;
    
    OntologyConcept::Relation rel;
    rel.type = type;
    rel.target = to;
    rel.weight = weight;
    rel.temporal_constraint = temporal;
    
    from_concept.addRelation(rel);
    
    return updateConcept(from_concept);
}

std::vector<OntologyConcept> OntologyGraph::getRelatedConcepts(
    const ConceptId& concept_id, RelationType type, int depth) const {
    std::vector<OntologyConcept> results;
    std::set<ConceptId> visited;
    std::queue<std::pair<ConceptId, int>> queue;
    
    queue.push({concept_id, 0});
    visited.insert(concept_id);
    
    while (!queue.empty()) {
        auto [current_id, current_depth] = queue.front();
        queue.pop();
        
        if (current_depth >= depth) continue;
        
        auto concept_opt = getConcept(current_id);
        if (!concept_opt) continue;
        
        auto concept = *concept_opt;
        
        for (const auto& rel : concept.relations) {
            if (rel.type == type && visited.find(rel.target) == visited.end()) {
                auto related_opt = getConcept(rel.target);
                if (related_opt) {
                    results.push_back(*related_opt);
                    visited.insert(rel.target);
                    queue.push({rel.target, current_depth + 1});
                }
            }
        }
    }
    
    return results;
}

std::vector<OntologyConcept> OntologyGraph::expandSemantically(
    const ConceptId& concept_id, int depth, float min_weight) const {
    std::vector<OntologyConcept> results;
    std::set<ConceptId> visited;
    std::queue<std::pair<ConceptId, int>> queue;
    
    queue.push({concept_id, 0});
    visited.insert(concept_id);
    
    while (!queue.empty()) {
        auto [current_id, current_depth] = queue.front();
        queue.pop();
        
        if (current_depth >= depth) continue;
        
        auto concept_opt = getConcept(current_id);
        if (!concept_opt) continue;
        
        auto concept = *concept_opt;
        
        for (const auto& rel : concept.relations) {
            if (rel.weight >= min_weight && 
                visited.find(rel.target) == visited.end()) {
                auto related_opt = getConcept(rel.target);
                if (related_opt) {
                    results.push_back(*related_opt);
                    visited.insert(rel.target);
                    queue.push({rel.target, current_depth + 1});
                }
            }
        }
    }
    
    return results;
}

void OntologyGraph::bindMemoryToConcept(const ConceptId& concept, 
                                       const MemoryId& memory) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto concept_opt = getConcept(concept);
    if (!concept_opt) return;
    
    auto c = *concept_opt;
    c.bindMemory(memory);
    updateConcept(c);
}

size_t OntologyGraph::getConceptCount() const {
    std::string countStr;
    pimpl_->db_->Get(rocksdb::ReadOptions(), 
                      pimpl_->metaKey("concept_count"), &countStr);
    return countStr.empty() ? 0 : std::stoul(countStr);
}

std::string OntologyGraph::exportToJson() const {
    json result;
    result["concepts"] = json::array();
    result["export_time"] = timestampToString(std::chrono::system_clock::now());
    
    // 使用迭代器遍历所有概念
    rocksdb::ReadOptions read_options;
    std::unique_ptr<rocksdb::Iterator> it(pimpl_->db_->NewIterator(read_options));
    
    for (it->Seek("c:"); it->Valid() && it->key().starts_with("c:"); it->Next()) {
        auto concept = OntologyConcept::fromJson(it->value().ToString());
        result["concepts"].push_back(json::parse(concept.toJson()));
    }
    
    return result.dump(2);
}

} // namespace pos
