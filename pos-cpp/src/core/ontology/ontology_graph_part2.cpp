/**
 * @file ontology_graph_part2.cpp
 * @brief 本体图谱实现 - 第二部分 (查询、遍历、推理)
 */

#include "ontology_graph.h"
#include <folly/json.h>
#include <queue>
#include <stack>
#include <set>
#include <math>

namespace personal_ontology {
namespace ontology {

// =============================================================================
// 查询操作
// =============================================================================

Result<std::vector<OntologyConcept>> OntologyGraph::findConceptsByName(
    const std::string& name, bool include_aliases) const {
    
    std::vector<OntologyConcept> results;
    
    // 使用名称索引查找
    auto ids = lookupByName(name);
    
    for (const auto& id : ids) {
        auto result = getConcept(id);
        if (result.isOk()) {
            results.push_back(result.value());
        }
    }
    
    // 如果包含别名，搜索别名
    if (include_aliases) {
        rocksdb::ReadOptions read_options;
        std::unique_ptr<rocksdb::Iterator> it(db_>NewIterator(read_options, cf_concepts_));
        
        for (it->SeekToFirst(); it->Valid(); it->Next()) {
            auto concept_result = deserializeConcept(it->value().ToString());
            if (concept_result.isOk()) {
                const auto& concept = concept_result.value();
                // 检查别名
                for (const auto& alias : concept.aliases) {
                    if (alias.find(name) != std::string::npos ||
                        name.find(alias) != std::string::npos) {
                        // 避免重复
                        bool already_in = false;
                        for (const auto& existing : results) {
                            if (existing.id == concept.id) {
                                already_in = true;
                                break;
                            }
                        }
                        if (!already_in) {
                            results.push_back(concept);
                        }
                        break;
                    }
                }
            }
        }
    }
    
    return Result<std::vector<OntologyConcept>>(std::move(results));
}

Result<std::vector<SemanticSimilarityResult>> OntologyGraph::semanticSearch(
    const Embedding& query_embedding, 
    size_t top_k,
    std::optional<ConceptType> type_filter) const {
    
    std::vector<SemanticSimilarityResult> results;
    
    // 遍历所有概念计算相似度
    rocksdb::ReadOptions read_options;
    std::unique_ptr<rocksdb::Iterator> it(db_>NewIterator(read_options, cf_concepts_));
    
    for (it->SeekToFirst(); it->Valid(); it->Next()) {
        auto concept_result = deserializeConcept(it->value().ToString());
        if (concept_result.isError()) continue;
        
        const auto& concept = concept_result.value();
        
        // 类型过滤
        if (type_filter.has_value() && concept.concept_type != type_filter.value()) {
            continue;
        }
        
        // 计算相似度
        if (concept.embedding.has_value()) {
            double similarity = cosineSimilarity(query_embedding, concept.embedding.value());
            
            if (results.size() < top_k) {
                SemanticSimilarityResult res;
                res.concept_id = concept.id;
                res.concept_name = concept.name;
                res.similarity_score = similarity;
                res.type = concept.concept_type;
                results.push_back(res);
                
                // 维护最小堆
                std::sort(results.begin(), results.end(),
                    [](const SemanticSimilarityResult& a, const SemanticSimilarityResult& b) {
                        return a.similarity_score > b.similarity_score;
                    });
            } else if (similarity > results.back().similarity_score) {
                results.back() = SemanticSimilarityResult{
                    concept.id, concept.name, similarity, concept.concept_type
                };
                std::sort(results.begin(), results.end(),
                    [](const SemanticSimilarityResult& a, const SemanticSimilarityResult& b) {
                        return a.similarity_score > b.similarity_score;
                    });
            }
        }
    }
    
    return Result<std::vector<SemanticSimilarityResult>>(std::move(results));
}

Result<std::vector<OntologyConcept>> OntologyGraph::searchConcepts(
    const std::string& keyword, size_t limit) const {
    
    std::vector<OntologyConcept> results;
    std::set<ConceptId> seen;
    
    rocksdb::ReadOptions read_options;
    std::unique_ptr<rocksdb::Iterator> it(db_>NewIterator(read_options, cf_concepts_));
    
    for (it->SeekToFirst(); it->Valid() && results.size() < limit; it->Next()) {
        auto concept_result = deserializeConcept(it->value().ToString());
        if (concept_result.isError()) continue;
        
        const auto& concept = concept_result.value();
        
        // 检查名称
        if (concept.name.find(keyword) != std::string::npos) {
            if (seen.insert(concept.id).second) {
                results.push_back(concept);
                continue;
            }
        }
        
        // 检查描述
        if (concept.description.find(keyword) != std::string::npos) {
            if (seen.insert(concept.id).second) {
                results.push_back(concept);
                continue;
            }
        }
        
        // 检查别名
        for (const auto& alias : concept.aliases) {
            if (alias.find(keyword) != std::string::npos) {
                if (seen.insert(concept.id).second) {
                    results.push_back(concept);
                }
                break;
            }
        }
    }
    
    return Result<std::vector<OntologyConcept>>(std::move(results));
}

Result<std::vector<ConceptId>> OntologyGraph::getAllConceptIds(
    size_t limit, size_t offset) const {
    
    std::vector<ConceptId> ids;
    ids.reserve(limit);
    
    rocksdb::ReadOptions read_options;
    std::unique_ptr<rocksdb::Iterator> it(db_>NewIterator(read_options, cf_concepts_));
    
    size_t skipped = 0;
    for (it->SeekToFirst(); it->Valid() && ids.size() < limit; it->Next()) {
        if (skipped < offset) {
            skipped++;
            continue;
        }
        ids.push_back(it->key().ToString());
    }
    
    return Result<std::vector<ConceptId>>(std::move(ids));
}

// =============================================================================
// 图谱遍历和推理
// =============================================================================

Result<bool> OntologyGraph::traverseDepthFirst(
    const ConceptId& start_id,
    GraphTraverser& traverser,
    int max_depth) const {
    
    auto start_result = getConcept(start_id);
    if (start_result.isError()) {
        return Result<bool>(start_result.errorCode(), start_result.errorMessage());
    }
    
    std::stack<std::pair<ConceptId, int>> stack;
    std::set<ConceptId> visited;
    
    stack.push({start_id, 0});
    
    while (!stack.empty()) {
        auto [current_id, depth] = stack.top();
        stack.pop();
        
        if (visited.count(current_id) > 0 || depth > max_depth) {
            continue;
        }
        visited.insert(current_id);
        
        auto concept_result = getConcept(current_id);
        if (concept_result.isError()) continue;
        
        const auto& concept = concept_result.value();
        
        // 访问节点
        if (!traverser.visitNode(concept, depth)) {
            return Result<bool>(true);  // 遍历被用户终止
        }
        
        // 遍历出边
        for (const auto& rel : concept.outgoing_relations) {
            if (traverser.shouldTraverse(rel, depth)) {
                if (!traverser.visitEdge(current_id, rel, depth)) {
                    return Result<bool>(true);
                }
                if (visited.count(rel.target_concept_id) == 0) {
                    stack.push({rel.target_concept_id, depth + 1});
                }
            }
        }
    }
    
    return Result<bool>(true);
}

Result<bool> OntologyGraph::traverseBreadthFirst(
    const ConceptId& start_id,
    GraphTraverser& traverser,
    int max_depth) const {
    
    auto start_result = getConcept(start_id);
    if (start_result.isError()) {
        return Result<bool>(start_result.errorCode(), start_result.errorMessage());
    }
    
    std::queue<std::pair<ConceptId, int>> queue;
    std::set<ConceptId> visited;
    
    queue.push({start_id, 0});
    
    while (!queue.empty()) {
        auto [current_id, depth] = queue.front();
        queue.pop();
        
        if (visited.count(current_id) > 0 || depth > max_depth) {
            continue;
        }
        visited.insert(current_id);
        
        auto concept_result = getConcept(current_id);
        if (concept_result.isError()) continue;
        
        const auto& concept = concept_result.value();
        
        // 访问节点
        if (!traverser.visitNode(concept, depth)) {
            return Result<bool>(true);
        }
        
        // 遍历出边
        for (const auto& rel : concept.outgoing_relations) {
            if (traverser.shouldTraverse(rel, depth)) {
                if (!traverser.visitEdge(current_id, rel, depth)) {
                    return Result<bool>(true);
                }
                if (visited.count(rel.target_concept_id) == 0) {
                    queue.push({rel.target_concept_id, depth + 1});
                }
            }
        }
    }
    
    return Result<bool>(true);
}

Result<std::vector<ConceptRelation>> OntologyGraph::findPath(
    const ConceptId& from, const ConceptId& to, int max_hops) const {
    
    if (!conceptExists(from)) {
        return Result<std::vector<ConceptRelation>>(
            ErrorCode::STORAGE_NOT_FOUND, "Source concept not found");
    }
    if (!conceptExists(to)) {
        return Result<std::vector<ConceptRelation>>(
            ErrorCode::STORAGE_NOT_FOUND, "Target concept not found");
    }
    
    // BFS找最短路径
    std::queue<std::pair<ConceptId, std::vector<ConceptRelation>>> queue;
    std::set<ConceptId> visited;
    
    queue.push({from, {}});
    visited.insert(from);
    
    while (!queue.empty()) {
        auto [current_id, path] = queue.front();
        queue.pop();
        
        if (path.size() >= static_cast<size_t>(max_hops)) {
            continue;
        }
        
        auto concept_result = getConcept(current_id);
        if (concept_result.isError()) continue;
        
        const auto& concept = concept_result.value();
        
        for (const auto& rel : concept.outgoing_relations) {
            std::vector<ConceptRelation> new_path = path;
            new_path.push_back(rel);
            
            if (rel.target_concept_id == to) {
                return Result<std::vector<ConceptRelation>>(std::move(new_path));
            }
            
            if (visited.count(rel.target_concept_id) == 0) {
                visited.insert(rel.target_concept_id);
                queue.push({rel.target_concept_id, new_path});
            }
        }
    }
    
    return Result<std::vector<ConceptRelation>>(
        ErrorCode::STORAGE_NOT_FOUND, "No path found");
}

Result<std::vector<OntologyConcept>> OntologyGraph::getSubConcepts(
    const ConceptId& concept_id, bool recursive) const {
    
    std::vector<OntologyConcept> results;
    std::set<ConceptId> seen;
    
    std::function<void(const ConceptId&, int)> collect = [&](const ConceptId& id, int depth) {
        if (!recursive && depth > 0) return;
        
        // 查找IS_A关系入边
        rocksdb::ReadOptions read_options;
        std::unique_ptr<rocksdb::Iterator> it(db_>NewIterator(read_options, cf_concepts_));
        
        for (it->SeekToFirst(); it->Valid(); it->Next()) {
            auto concept_result = deserializeConcept(it->value().ToString());
            if (concept_result.isError()) continue;
            
            const auto& concept = concept_result.value();
            
            for (const auto& rel : concept.outgoing_relations) {
                if (rel.target_concept_id == id && rel.type == RelationType::IS_A) {
                    if (seen.insert(concept.id).second) {
                        results.push_back(concept);
                        if (recursive) {
                            collect(concept.id, depth + 1);
                        }
                    }
                    break;
                }
            }
        }
    };
    
    collect(concept_id, 0);
    return Result<std::vector<OntologyConcept>>(std::move(results));
}

Result<std::vector<OntologyConcept>> OntologyGraph::getSuperConcepts(
    const ConceptId& concept_id, bool recursive) const {
    
    auto concept_result = getConcept(concept_id);
    if (concept_result.isError()) {
        return Result<std::vector<OntologyConcept>>(
            concept_result.errorCode(), concept_result.errorMessage());
    }
    
    std::vector<OntologyConcept> results;
    std::set<ConceptId> seen;
    
    std::function<void(const ConceptId&, int)> collect = [&](const ConceptId& id, int depth) {
        if (!recursive && depth > 0) return;
        
        auto result = getConcept(id);
        if (result.isError()) return;
        
        const auto& concept = result.value();
        
        for (const auto& rel : concept.outgoing_relations) {
            if (rel.type == RelationType::IS_A) {
                auto super_result = getConcept(rel.target_concept_id);
                if (super_result.isOk()) {
                    const auto& super = super_result.value();
                    if (seen.insert(super.id).second) {
                        results.push_back(super);
                        if (recursive) {
                            collect(super.id, depth + 1);
                        }
                    }
                }
            }
        }
    };
    
    collect(concept_id, 0);
    return Result<std::vector<OntologyConcept>>(std::move(results));
}

Result<double> OntologyGraph::calculateSemanticDistance(
    const ConceptId& id1, const ConceptId& id2) const {
    
    auto concept1_result = getConcept(id1);
    auto concept2_result = getConcept(id2);
    
    if (concept1_result.isError()) {
        return Result<double>(concept1_result.errorCode(), concept1_result.errorMessage());
    }
    if (concept2_result.isError()) {
        return Result<double>(concept2_result.errorCode(), concept2_result.errorMessage());
    }
    
    const auto& concept1 = concept1_result.value();
    const auto& concept2 = concept2_result.value();
    
    // 如果都有向量嵌入，使用向量距离
    if (concept1.embedding.has_value() && concept2.embedding.has_value()) {
        double dist = 1.0 - cosineSimilarity(concept1.embedding.value(), concept2.embedding.value());
        return Result<double>(dist);
    }
    
    // 否则使用图距离
    auto path_result = findPath(id1, id2, 10);
    if (path_result.isOk()) {
        // 路径越短，距离越小
        double distance = path_result.value().size() / 10.0;
        return Result<double>(distance);
    }
    
    // 无法计算距离
    return Result<double>(1.0);
}

// =============================================================================
// 关系推理
// =============================================================================

Result<std::vector<ConceptRelation>> OntologyGraph::inferRelations(
    const ConceptId& concept_id, int depth) const {
    
    std::vector<ConceptRelation> inferred;
    std::set<std::pair<ConceptId, RelationType>> known_relations;
    
    // 收集已知关系
    auto concept_result = getConcept(concept_id);
    if (concept_result.isError()) {
        return Result<std::vector<ConceptRelation>>(
            concept_result.errorCode(), concept_result.errorMessage());
    }
    
    const auto& concept = concept_result.value();
    for (const auto& rel : concept.outgoing_relations) {
        known_relations.insert({rel.target_concept_id, rel.type});
    }
    
    // 1. 传递性推理 (A is_a B, B is_a C => A is_a C)
    for (const auto& rel : concept.outgoing_relations) {
        if (rel.type == RelationType::IS_A) {
            auto super_result = getConcept(rel.target_concept_id);
            if (super_result.isOk()) {
                const auto& super = super_result.value();
                for (const auto& super_rel : super.outgoing_relations) {
                    if (super_rel.type == RelationType::IS_A) {
                        // 推断传递关系
                        if (known_relations.count({super_rel.target_concept_id, RelationType::IS_A}) == 0) {
                            ConceptRelation inferred_rel;
                            inferred_rel.type = RelationType::IS_A;
                            inferred_rel.target_concept_id = super_rel.target_concept_id;
                            inferred_rel.confidence = rel.confidence * super_rel.confidence * 0.9;
                            inferred.push_back(inferred_rel);
                        }
                    }
                }
            }
        }
    }
    
    // 2. 对称性推理 (A related_to B => B related_to A)
    // 通过入边反向推导
    
    // 3. 属性继承 (A is_a B, B has_property P => A has_property P)
    for (const auto& rel : concept.outgoing_relations) {
        if (rel.type == RelationType::IS_A) {
            auto super_result = getConcept(rel.target_concept_id);
            if (super_result.isOk()) {
                const auto& super = super_result.value();
                for (const auto& super_rel : super.outgoing_relations) {
                    if (super_rel.type == RelationType::HAS_PROPERTY) {
                        if (known_relations.count({super_rel.target_concept_id, RelationType::HAS_PROPERTY}) == 0) {
                            ConceptRelation inferred_rel = super_rel;
                            inferred_rel.confidence *= 0.95;
                            inferred.push_back(inferred_rel);
                        }
                    }
                }
            }
        }
    }
    
    return Result<std::vector<ConceptRelation>>(std::move(inferred));
}

Result<std::vector<OntologyConcept>> OntologyGraph::inferSimilarConcepts(
    const ConceptId& concept_id, double threshold) const {
    
    std::vector<OntologyConcept> similar;
    std::set<ConceptId> seen;
    seen.insert(concept_id);
    
    auto concept_result = getConcept(concept_id);
    if (concept_result.isError()) {
        return Result<std::vector<OntologyConcept>>(
            concept_result.errorCode(), concept_result.errorMessage());
    }
    
    const auto& concept = concept_result.value();
    
    // 1. SIMILAR_TO关系
    for (const auto& rel : concept.outgoing_relations) {
        if (rel.type == RelationType::SIMILAR_TO && rel.confidence >= threshold) {
            if (seen.insert(rel.target_concept_id).second) {
                auto target_result = getConcept(rel.target_concept_id);
                if (target_result.isOk()) {
                    similar.push_back(target_result.value());
                }
            }
        }
    }
    
    // 2. 入边SIMILAR_TO
    for (const auto& rel : concept.incoming_relations) {
        if (rel.type == RelationType::SIMILAR_TO && rel.confidence >= threshold) {
            // 需要找到源概念
            rocksdb::ReadOptions read_options;
            std::unique_ptr<rocksdb::Iterator> it(db_>NewIterator(read_options, cf_concepts_));
            
            for (it->SeekToFirst(); it->Valid(); it->Next()) {
                auto other_result = deserializeConcept(it->value().ToString());
                if (other_result.isError()) continue;
                
                const auto& other = other_result.value();
                for (const auto& other_rel : other.outgoing_relations) {
                    if (other_rel.target_concept_id == concept_id &&
                        other_rel.type == RelationType::SIMILAR_TO) {
                        if (seen.insert(other.id).second) {
                            similar.push_back(other);
                        }
                        break;
                    }
                }
            }
        }
    }
    
    return Result<std::vector<OntologyConcept>>(std::move(similar));
}

Result<std::vector<OntologyConcept>> OntologyGraph::findRelatedConcepts(
    const ConceptId& concept_id, 
    std::optional<RelationType> relation_type,
    int max_depth) const {
    
    std::vector<OntologyConcept> related;
    std::set<ConceptId> seen;
    seen.insert(concept_id);
    
    std::queue<std::pair<ConceptId, int>> queue;
    queue.push({concept_id, 0});
    
    while (!queue.empty()) {
        auto [current_id, depth] = queue.front();
        queue.pop();
        
        if (depth >= max_depth) continue;
        
        auto concept_result = getConcept(current_id);
        if (concept_result.isError()) continue;
        
        const auto& concept = concept_result.value();
        
        for (const auto& rel : concept.outgoing_relations) {
            if (!relation_type.has_value() || rel.type == relation_type.value()) {
                if (seen.insert(rel.target_concept_id).second) {
                    auto target_result = getConcept(rel.target_concept_id);
                    if (target_result.isOk()) {
                        related.push_back(target_result.value());
                        queue.push({rel.target_concept_id, depth + 1});
                    }
                }
            }
        }
    }
    
    return Result<std::vector<OntologyConcept>>(std::move(related));
}

// =============================================================================
// 高级功能
// =============================================================================

Result<bool> OntologyGraph::mergeConcepts(
    const ConceptId& keep_id,
    const ConceptId& merge_id,
    bool merge_relations) {
    
    auto keep_result = getConcept(keep_id);
    auto merge_result = getConcept(merge_id);
    
    if (keep_result.isError()) {
        return Result<bool>(keep_result.errorCode(), keep_result.errorMessage());
    }
    if (merge_result.isError()) {
        return Result<bool>(merge_result.errorCode(), merge_result.errorMessage());
    }
    
    auto keep = keep_result.value();
    const auto& merge = merge_result.value();
    
    // 合并别名
    for (const auto& alias : merge.aliases) {
        if (std::find(keep.aliases.begin(), keep.aliases.end(), alias) == keep.aliases.end()) {
            keep.aliases.push_back(alias);
        }
    }
    
    // 合并定义
    for (const auto& def : merge.definitions) {
        if (std::find(keep.definitions.begin(), keep.definitions.end(), def) == keep.definitions.end()) {
            keep.definitions.push_back(def);
        }
    }
    
    // 合并来源
    for (const auto& source : merge.sources) {
        if (std::find(keep.sources.begin(), keep.sources.end(), source) == keep.sources.end()) {
            keep.sources.push_back(source);
        }
    }
    
    // 合并关系
    if (merge_relations) {
        for (const auto& rel : merge.outgoing_relations) {
            if (rel.target_concept_id != keep_id) {  // 避免自环
                bool exists = false;
                for (const auto& existing : keep.outgoing_relations) {
                    if (existing.target_concept_id == rel.target_concept_id &&
                        existing.type == rel.type) {
                        exists = true;
                        break;
                    }
                }
                if (!exists) {
                    keep.outgoing_relations.push_back(rel);
                }
            }
        }
    }
    
    // 更新引用计数
    keep.reference_count += merge.reference_count;
    
    // 保存更新后的概念
    rocksdb::WriteOptions write_options;
    auto status = db_>Put(write_options, cf_concepts_, keep_id, serializeConcept(keep));
    
    if (!status.ok()) {
        return Result<bool>(ErrorCode::STORAGE_WRITE_ERROR,
            std::format("Failed to update concept: {}", status.ToString()));
    }
    
    // 删除被合并的概念
    auto delete_result = deleteConcept(merge_id);
    if (delete_result.isError()) {
        return delete_result;
    }
    
    // 更新缓存
    {
        tbb::concurrent_hash_map<ConceptId, OntologyConcept>::accessor accessor;
        if (concept_cache_.find(accessor, keep_id)) {
            accessor->second = keep;
        }
    }
    concept_cache_.erase(merge_id);
    
    return Result<bool>(true);
}

Result<ConceptId> OntologyGraph::splitConcept(
    const ConceptId& source_id,
    OntologyConcept new_concept,
    const std::vector<ConceptRelation>& relations_to_move) {
    
    auto source_result = getConcept(source_id);
    if (source_result.isError()) {
        return Result<ConceptId>(source_result.errorCode(), source_result.errorMessage());
    }
    
    auto source = source_result.value();
    
    // 创建新概念
    if (new_concept.id.empty()) {
        new_concept.id = generateUUID();
    }
    new_concept.created_at = MemoryTrace::now();
    new_concept.updated_at = new_concept.created_at;
    
    // 转移关系
    for (const auto& rel : relations_to_move) {
        new_concept.outgoing_relations.push_back(rel);
        
        // 从源概念移除
        source.outgoing_relations.erase(
            std::remove_if(source.outgoing_relations.begin(), source.outgoing_relations.end(),
                [&rel](const ConceptRelation& r) {
                    return r.target_concept_id == rel.target_concept_id &&
                           r.type == rel.type;
                }),
            source.outgoing_relations.end()
        );
    }
    
    // 添加IS_A关系到源概念 (新概念是源概念的子类)
    ConceptRelation is_a_rel;
    is_a_rel.type = RelationType::IS_A;
    is_a_rel.target_concept_id = source_id;
    is_a_rel.confidence = 1.0;
    is_a_rel.created_at = MemoryTrace::now();
    new_concept.outgoing_relations.push_back(is_a_rel);
    
    // 存储两个概念
    rocksdb::WriteBatch batch;
    batch.Put(cf_concepts_, source_id, serializeConcept(source));
    batch.Put(cf_concepts_, new_concept.id, serializeConcept(new_concept));
    
    rocksdb::WriteOptions write_options;
    auto status = db_>Write(write_options, &batch);
    
    if (!status.ok()) {
        return Result<ConceptId>(ErrorCode::STORAGE_WRITE_ERROR,
            std::format("Failed to split concept: {}", status.ToString()));
    }
    
    // 更新缓存
    {
        tbb::concurrent_hash_map<ConceptId, OntologyConcept>::accessor accessor;
        if (concept_cache_.find(accessor, source_id)) {
            accessor->second = source;
        }
        if (concept_cache_.insert(accessor, new_concept.id)) {
            accessor->second = new_concept;
        }
    }
    
    return Result<ConceptId>(new_concept.id);
}

Result<OntologyGraph::ImportStats> OntologyGraph::importConcepts(
    const std::vector<OntologyConcept>& concepts,
    const std::string& source,
    bool skip_existing) {
    
    ImportStats stats;
    
    for (const auto& concept : concepts) {
        // 检查是否已存在
        if (conceptExists(concept.id)) {
            if (skip_existing) {
                stats.skipped++;
                continue;
            }
            
            // 更新现有概念
            auto existing_result = getConcept(concept.id);
            if (existing_result.isOk()) {
                auto existing = existing_result.value();
                
                // 合并信息
                ConceptUpdateRequest update;
                update.add_sources = {source};
                
                auto update_result = updateConcept(concept.id, update);
                if (update_result.isOk()) {
                    stats.updated++;
                } else {
                    stats.failed++;
                }
                continue;
            }
        }
        
        // 创建新概念
        auto new_concept = concept;
        if (std::find(new_concept.sources.begin(), new_concept.sources.end(), source) 
            == new_concept.sources.end()) {
            new_concept.sources.push_back(source);
        }
        
        auto create_result = createConcept(new_concept);
        if (create_result.isOk()) {
            stats.created++;
        } else {
            stats.failed++;
        }
    }
    
    return Result<ImportStats>(stats);
}

Result<std::vector<OntologyConcept>> OntologyGraph::exportConcepts(
    const std::vector<ConceptId>& concept_ids,
    bool include_relations) const {
    
    std::vector<OntologyConcept> results;
    
    if (concept_ids.empty()) {
        // 导出所有概念
        auto all_ids = getAllConceptIds(100000, 0);
        if (all_ids.isError()) {
            return Result<std::vector<OntologyConcept>>(
                all_ids.errorCode(), all_ids.errorMessage());
        }
        return getConceptsBatch(all_ids.value());
    } else {
        return getConceptsBatch(concept_ids);
    }
}
