/**
 * @file ontology_graph_part3.cpp
 * @brief 本体图谱实现 - 第三部分 (维护、统计、辅助函数)
 */

#include "ontology_graph.h"
#include <folly/json.h>
#include <filesystem>
#include <math>

namespace personal_ontology {
namespace ontology {

// =============================================================================
// 维护和统计
// =============================================================================

GraphStatistics OntologyGraph::getStatistics() const {
    GraphStatistics stats;
    
    rocksdb::ReadOptions read_options;
    
    // 统计概念
    std::unique_ptr<rocksdb::Iterator> concept_it(db_>NewIterator(read_options, cf_concepts_));
    for (concept_it->SeekToFirst(); concept_it->Valid(); concept_it->Next()) {
        stats.concept_count++;
        
        auto concept_result = deserializeConcept(concept_it->value().ToString());
        if (concept_result.isOk()) {
            const auto& concept = concept_result.value();
            size_t type_idx = static_cast<size_t>(concept.concept_type);
            if (type_idx < 9) {
                stats.type_distribution[type_idx]++;
            }
        }
    }
    
    // 统计关系
    std::unique_ptr<rocksdb::Iterator> relation_it(db_>NewIterator(read_options, cf_relations_));
    for (relation_it->SeekToFirst(); relation_it->Valid(); relation_it->Next()) {
        stats.relation_count++;
        
        auto rel_result = deserializeRelation(relation_it->value().ToString());
        if (rel_result.isOk()) {
            const auto& rel = rel_result.value();
            stats.relation_distribution[rel.type]++;
        }
    }
    
    // 获取存储统计
    std::string db_stats;
    db_>GetProperty("rocksdb.estimate-num-keys", &db_stats);
    
    return stats;
}

Result<std::vector<std::string>> OntologyGraph::checkIntegrity(bool auto_fix) {
    std::vector<std::string> issues;
    
    rocksdb::ReadOptions read_options;
    std::unique_ptr<rocksdb::Iterator> it(db_>NewIterator(read_options, cf_concepts_));
    
    for (it->SeekToFirst(); it->Valid(); it->Next()) {
        auto concept_result = deserializeConcept(it->value().ToString());
        if (concept_result.isError()) {
            issues.push_back(std::format("Corrupted concept: {}", it->key().ToString()));
            if (auto_fix) {
                rocksdb::WriteOptions write_options;
                db_>Delete(write_options, cf_concepts_, it->key());
            }
            continue;
        }
        
        const auto& concept = concept_result.value();
        
        // 检查关系目标是否存在
        for (const auto& rel : concept.outgoing_relations) {
            if (!conceptExists(rel.target_concept_id)) {
                issues.push_back(std::format("Dangling relation from {} to {}",
                    concept.id, rel.target_concept_id));
                if (auto_fix) {
                    // 移除无效关系
                    removeRelation(concept.id, rel.target_concept_id, rel.type);
                }
            }
        }
    }
    
    return Result<std::vector<std::string>>(std::move(issues));
}

Result<bool> OntologyGraph::compact() {
    rocksdb::CompactRangeOptions compact_options;
    
    auto status = db_>CompactRange(compact_options, cf_concepts_, nullptr, nullptr);
    if (!status.ok()) {
        return Result<bool>(ErrorCode::INTERNAL_ERROR,
            std::format("Compact concepts failed: {}", status.ToString()));
    }
    
    status = db_>CompactRange(compact_options, cf_relations_, nullptr, nullptr);
    if (!status.ok()) {
        return Result<bool>(ErrorCode::INTERNAL_ERROR,
            std::format("Compact relations failed: {}", status.ToString()));
    }
    
    status = db_>CompactRange(compact_options, cf_index_, nullptr, nullptr);
    if (!status.ok()) {
        return Result<bool>(ErrorCode::INTERNAL_ERROR,
            std::format("Compact indices failed: {}", status.ToString()));
    }
    
    return Result<bool>(true);
}

Result<bool> OntologyGraph::createCheckpoint(const std::string& checkpoint_path) {
    try {
        std::filesystem::create_directories(checkpoint_path);
        
        rocksdb::Checkpoint* checkpoint = nullptr;
        auto status = rocksdb::Checkpoint::Create(db_.get(), &checkpoint);
        
        if (!status.ok()) {
            return Result<bool>(ErrorCode::INTERNAL_ERROR,
                std::format("Checkpoint creation failed: {}", status.ToString()));
        }
        
        status = checkpoint->CreateCheckpoint(checkpoint_path);
        delete checkpoint;
        
        if (!status.ok()) {
            return Result<bool>(ErrorCode::INTERNAL_ERROR,
                std::format("Checkpoint failed: {}", status.ToString()));
        }
        
        return Result<bool>(true);
        
    } catch (const std::exception& e) {
        return Result<bool>(ErrorCode::INTERNAL_ERROR,
            std::format("Checkpoint error: {}", e.what()));
    }
}

// =============================================================================
// 序列化
// =============================================================================

std::string OntologyGraph::serializeConcept(const OntologyConcept& concept) const {
    folly::dynamic obj = folly::dynamic::object;
    
    obj["id"] = concept.id;
    obj["name"] = concept.name;
    obj["concept_type"] = static_cast<int>(concept.concept_type);
    obj["description"] = concept.description;
    obj["confidence"] = concept.confidence;
    obj["created_at"] = concept.created_at;
    obj["updated_at"] = concept.updated_at;
    obj["reference_count"] = concept.reference_count;
    
    // 别名
    obj["aliases"] = folly::dynamic::array;
    for (const auto& alias : concept.aliases) {
        obj["aliases"].push_back(alias);
    }
    
    // 定义
    obj["definitions"] = folly::dynamic::array;
    for (const auto& def : concept.definitions) {
        obj["definitions"].push_back(def);
    }
    
    // 来源
    obj["sources"] = folly::dynamic::array;
    for (const auto& source : concept.sources) {
        obj["sources"].push_back(source);
    }
    
    // 向量嵌入
    if (concept.embedding.has_value()) {
        folly::dynamic emb = folly::dynamic::array;
        for (float val : concept.embedding.value()) {
            emb.push_back(val);
        }
        obj["embedding"] = emb;
    }
    
    // 出边关系
    obj["outgoing_relations"] = folly::dynamic::array;
    for (const auto& rel : concept.outgoing_relations) {
        folly::dynamic rel_obj = folly::dynamic::object
            ("type", static_cast<int>(rel.type))
            ("target_concept_id", rel.target_concept_id)
            ("confidence", rel.confidence)
            ("created_at", rel.created_at);
        
        if (rel.custom_name.has_value()) {
            rel_obj["custom_name"] = rel.custom_name.value();
        }
        
        obj["outgoing_relations"].push_back(rel_obj);
    }
    
    // 入边关系
    obj["incoming_relations"] = folly::dynamic::array;
    for (const auto& rel : concept.incoming_relations) {
        folly::dynamic rel_obj = folly::dynamic::object
            ("type", static_cast<int>(rel.type))
            ("target_concept_id", rel.target_concept_id)
            ("confidence", rel.confidence)
            ("created_at", rel.created_at);
        
        if (rel.custom_name.has_value()) {
            rel_obj["custom_name"] = rel.custom_name.value();
        }
        
        obj["incoming_relations"].push_back(rel_obj);
    }
    
    // 属性
    obj["properties"] = folly::dynamic::object;
    for (const auto& [key, value] : concept.properties) {
        // 简化属性序列化
        std::visit([&obj, &key](auto&& arg) {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, bool>) {
                obj["properties"][key] = arg;
            } else if constexpr (std::is_same_v<T, int64_t>) {
                obj["properties"][key] = arg;
            } else if constexpr (std::is_same_v<T, double>) {
                obj["properties"][key] = arg;
            } else if constexpr (std::is_same_v<T, std::string>) {
                obj["properties"][key] = arg;
            }
        }, value);
    }
    
    return folly::toJson(obj);
}

Result<OntologyConcept> OntologyGraph::deserializeConcept(const std::string& data) const {
    try {
        auto obj = folly::parseJson(data);
        OntologyConcept concept;
        
        concept.id = obj["id"].asString();
        concept.name = obj["name"].asString();
        concept.concept_type = static_cast<ConceptType>(obj["concept_type"].asInt());
        concept.description = obj["description"].asString();
        concept.confidence = obj["confidence"].asDouble();
        concept.created_at = obj["created_at"].asInt();
        concept.updated_at = obj["updated_at"].asInt();
        concept.reference_count = obj["reference_count"].asInt();
        
        // 别名
        for (const auto& alias : obj["aliases"]) {
            concept.aliases.push_back(alias.asString());
        }
        
        // 定义
        for (const auto& def : obj["definitions"]) {
            concept.definitions.push_back(def.asString());
        }
        
        // 来源
        for (const auto& source : obj["sources"]) {
            concept.sources.push_back(source.asString());
        }
        
        // 向量嵌入
        if (obj.find("embedding") != obj.items().end()) {
            Embedding emb;
            for (const auto& val : obj["embedding"]) {
                emb.push_back(static_cast<float>(val.asDouble()));
            }
            concept.embedding = emb;
        }
        
        // 出边关系
        for (const auto& rel_obj : obj["outgoing_relations"]) {
            ConceptRelation rel;
            rel.type = static_cast<RelationType>(rel_obj["type"].asInt());
            rel.target_concept_id = rel_obj["target_concept_id"].asString();
            rel.confidence = rel_obj["confidence"].asDouble();
            rel.created_at = rel_obj["created_at"].asInt();
            
            if (rel_obj.find("custom_name") != rel_obj.items().end()) {
                rel.custom_name = rel_obj["custom_name"].asString();
            }
            
            concept.outgoing_relations.push_back(rel);
        }
        
        // 入边关系
        for (const auto& rel_obj : obj["incoming_relations"]) {
            ConceptRelation rel;
            rel.type = static_cast<RelationType>(rel_obj["type"].asInt());
            rel.target_concept_id = rel_obj["target_concept_id"].asString();
            rel.confidence = rel_obj["confidence"].asDouble();
            rel.created_at = rel_obj["created_at"].asInt();
            
            if (rel_obj.find("custom_name") != rel_obj.items().end()) {
                rel.custom_name = rel_obj["custom_name"].asString();
            }
            
            concept.incoming_relations.push_back(rel);
        }
        
        return Result<OntologyConcept>(std::move(concept));
        
    } catch (const std::exception& e) {
        return Result<OntologyConcept>(ErrorCode::STORAGE_SERIALIZATION_ERROR,
            std::format("Deserialize failed: {}", e.what()));
    }
}

std::string OntologyGraph::serializeRelation(const ConceptRelation& relation) const {
    folly::dynamic obj = folly::dynamic::object
        ("type", static_cast<int>(relation.type))
        ("target_concept_id", relation.target_concept_id)
        ("confidence", relation.confidence)
        ("created_at", relation.created_at);
    
    if (relation.custom_name.has_value()) {
        obj["custom_name"] = relation.custom_name.value();
    }
    
    // 属性
    obj["attributes"] = folly::dynamic::object;
    
    return folly::toJson(obj);
}

Result<ConceptRelation> OntologyGraph::deserializeRelation(const std::string& data) const {
    try {
        auto obj = folly::parseJson(data);
        ConceptRelation relation;
        
        relation.type = static_cast<RelationType>(obj["type"].asInt());
        relation.target_concept_id = obj["target_concept_id"].asString();
        relation.confidence = obj["confidence"].asDouble();
        relation.created_at = obj["created_at"].asInt();
        
        if (obj.find("custom_name") != obj.items().end()) {
            relation.custom_name = obj["custom_name"].asString();
        }
        
        return Result<ConceptRelation>(std::move(relation));
        
    } catch (const std::exception& e) {
        return Result<ConceptRelation>(ErrorCode::STORAGE_SERIALIZATION_ERROR,
            std::format("Deserialize relation failed: {}", e.what()));
    }
}

// =============================================================================
// 索引管理
// =============================================================================

Result<bool> OntologyGraph::updateNameIndex(
    const ConceptId& id, const std::string& old_name, const std::string& new_name) {
    
    rocksdb::WriteBatch batch;
    
    // 删除旧索引
    if (!old_name.empty()) {
        std::string old_key = std::format("name:{}", old_name);
        batch.Delete(cf_index_, old_key);
    }
    
    // 添加新索引
    if (!new_name.empty()) {
        std::string new_key = std::format("name:{}", new_name);
        batch.Put(cf_index_, new_key, id);
    }
    
    rocksdb::WriteOptions write_options;
    auto status = db_>Write(write_options, &batch);
    
    if (!status.ok()) {
        return Result<bool>(ErrorCode::STORAGE_WRITE_ERROR,
            std::format("Index update failed: {}", status.ToString()));
    }
    
    return Result<bool>(true);
}

Result<bool> OntologyGraph::updateRelationIndices(
    const ConceptId& from, const ConceptRelation& relation, bool add) {
    
    // 关系类型索引
    std::string type_key = std::format("rel_type:{}:{}",
        static_cast<int>(relation.type), from);
    
    rocksdb::WriteOptions write_options;
    
    if (add) {
        db_>Put(write_options, cf_index_, type_key, relation.target_concept_id);
    } else {
        db_>Delete(write_options, cf_index_, type_key);
    }
    
    return Result<bool>(true);
}

std::vector<ConceptId> OntologyGraph::lookupByName(const std::string& name) const {
    std::vector<ConceptId> ids;
    
    rocksdb::ReadOptions read_options;
    std::string index_key = std::format("name:{}", name);
    std::string id;
    
    auto status = db_>Get(read_options, cf_index_, index_key, &id);
    if (status.ok()) {
        ids.push_back(id);
    }
    
    return ids;
}

// =============================================================================
// HNSW索引 (语义搜索)
// =============================================================================

void OntologyGraph::buildHnswIndex() {
    // HNSW索引构建逻辑
    // 遍历所有概念，将向量嵌入加入HNSW索引
}

// =============================================================================
// 工具函数
// =============================================================================

double OntologyGraph::cosineSimilarity(const Embedding& a, const Embedding& b) const {
    if (a.size() != b.size()) return 0.0;
    
    double dot = 0.0;
    double norm_a = 0.0;
    double norm_b = 0.0;
    
    for (size_t i = 0; i < a.size(); i++) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    
    if (norm_a == 0.0 || norm_b == 0.0) return 0.0;
    
    return dot / (std::sqrt(norm_a) * std::sqrt(norm_b));
}

} // namespace ontology
} // namespace personal_ontology
