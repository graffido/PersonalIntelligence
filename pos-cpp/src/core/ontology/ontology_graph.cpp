/**
 * @file ontology_graph.cpp
 * @brief 本体图谱实现
 * 
 * 实现概念CRUD、关系管理、语义搜索和图谱推理
 */

#include "ontology_graph.h"
#include <rocksdb/write_batch.h>
#include <rocksdb/snapshot.h>
#include <rocksdb/table.h>
#include <folly/json.h>
#include <queue>
#include <stack>
#include <set>
#include <filesystem>
#include <algorithm>

namespace personal_ontology {
namespace ontology {

// =============================================================================
// 构造函数和析构函数
// =============================================================================

OntologyGraph::OntologyGraph(const OntologyGraphConfig& config)
    : config_(config), initialized_(false) {
}

OntologyGraph::~OntologyGraph() {
    if (initialized_) {
        shutdown();
    }
}

OntologyGraph::OntologyGraph(OntologyGraph&& other) noexcept
    : config_(std::move(other.config_))
    , db_(std::move(other.db_))
    , cf_concepts_(other.cf_concepts_)
    , cf_relations_(other.cf_relations_)
    , cf_index_(other.cf_index_)
    , initialized_(other.initialized_)
    , concept_cache_(std::move(other.concept_cache_))
    , hnsw_index_(std::move(other.hnsw_index_)) {
    
    other.cf_concepts_ = nullptr;
    other.cf_relations_ = nullptr;
    other.cf_index_ = nullptr;
    other.initialized_ = false;
}

OntologyGraph& OntologyGraph::operator=(OntologyGraph&& other) noexcept {
    if (this != &other) {
        if (initialized_) {
            shutdown();
        }
        
        config_ = std::move(other.config_);
        db_ = std::move(other.db_);
        cf_concepts_ = other.cf_concepts_;
        cf_relations_ = other.cf_relations_;
        cf_index_ = other.cf_index_;
        initialized_ = other.initialized_;
        concept_cache_ = std::move(other.concept_cache_);
        hnsw_index_ = std::move(other.hnsw_index_);
        
        other.cf_concepts_ = nullptr;
        other.cf_relations_ = nullptr;
        other.cf_index_ = nullptr;
        other.initialized_ = false;
    }
    return *this;
}

// =============================================================================
// 初始化和关闭
// =============================================================================

Result<bool> OntologyGraph::initialize() {
    try {
        // 创建数据目录
        std::filesystem::create_directories(config_.data_path);
        
        // 配置RocksDB选项
        rocksdb::Options options;
        options.create_if_missing = config_.create_if_missing;
        options.max_open_files = config_.max_open_files;
        
        // 配置块缓存
        rocksdb::BlockBasedTableOptions table_options;
        table_options.block_cache = rocksdb::NewLRUCache(config_.cache_size_mb * 1024 * 1024);
        options.table_factory.reset(rocksdb::NewBlockBasedTableFactory(table_options));
        
        // 配置压缩
        if (config_.enable_compression) {
            options.compression = rocksdb::kLZ4Compression;
            options.bottommost_compression = rocksdb::kZSTD;
        }
        
        // 配置WAL
        options.WAL_ttl_seconds = 3600;  // 1小时WAL保留
        
        // 列族配置
        rocksdb::ColumnFamilyOptions cf_options;
        cf_options.compression = rocksdb::kLZ4Compression;
        
        std::vector<rocksdb::ColumnFamilyDescriptor> cf_descriptors;
        cf_descriptors.emplace_back("concepts", cf_options);
        cf_descriptors.emplace_back("relations", cf_options);
        cf_descriptors.emplace_back("indices", cf_options);
        
        std::vector<rocksdb::ColumnFamilyHandle*> handles;
        rocksdb::Status status;
        
        // 尝试打开现有数据库
        status = rocksdb::DB::Open(options, config_.data_path, cf_descriptors, &handles, &db_);
        
        if (!status.ok()) {
            // 可能是新数据库，尝试创建
            if (config_.create_if_missing) {
                status = rocksdb::DB::Open(options, config_.data_path, &db_);
                if (status.ok()) {
                    // 创建列族
                    for (const auto& desc : cf_descriptors) {
                        rocksdb::ColumnFamilyHandle* cf;
                        status = db_>CreateColumnFamily(desc.options, desc.name, &cf);
                        if (!status.ok()) {
                            return Result<bool>(ErrorCode::STORAGE_WRITE_ERROR,
                                std::format("Failed to create column family {}: {}", 
                                    desc.name, status.ToString()));
                        }
                        delete cf;
                    }
                    db_.reset();
                    
                    // 重新打开
                    status = rocksdb::DB::Open(options, config_.data_path, cf_descriptors, &handles, &db_);
                }
            }
        }
        
        if (!status.ok()) {
            return Result<bool>(ErrorCode::STORAGE_WRITE_ERROR,
                std::format("Failed to open database: {}", status.ToString()));
        }
        
        cf_concepts_ = handles[0];
        cf_relations_ = handles[1];
        cf_index_ = handles[2];
        
        initialized_ = true;
        return Result<bool>(true);
        
    } catch (const std::exception& e) {
        return Result<bool>(ErrorCode::INTERNAL_ERROR,
            std::format("Initialization failed: {}", e.what()));
    }
}

void OntologyGraph::shutdown() {
    if (!initialized_) return;
    
    // 刷新所有数据
    db_>Flush(rocksdb::FlushOptions());
    
    // 关闭列族句柄
    delete cf_concepts_;
    delete cf_relations_;
    delete cf_index_;
    
    cf_concepts_ = nullptr;
    cf_relations_ = nullptr;
    cf_index_ = nullptr;
    
    db_.reset();
    initialized_ = false;
}

// =============================================================================
// 概念CRUD操作
// =============================================================================

Result<ConceptId> OntologyGraph::createConcept(OntologyConcept concept) {
    if (concept.id.empty()) {
        concept.id = generateUUID();
    }
    concept.created_at = MemoryTrace::now();
    concept.updated_at = concept.created_at;
    
    // 检查是否已存在
    if (conceptExists(concept.id)) {
        return Result<ConceptId>(ErrorCode::STORAGE_ALREADY_EXISTS,
            std::format("Concept already exists: {}", concept.id));
    }
    
    // 序列化并存储
    std::string data = serializeConcept(concept);
    
    rocksdb::WriteOptions write_options;
    write_options.sync = config_.sync_writes;
    
    rocksdb::Status status = db_>Put(write_options, cf_concepts_, concept.id, data);
    
    if (!status.ok()) {
        return Result<ConceptId>(ErrorCode::STORAGE_WRITE_ERROR,
            std::format("Failed to store concept: {}", status.ToString()));
    }
    
    // 更新名称索引
    updateNameIndex(concept.id, "", concept.name);
    
    // 添加到缓存
    {
        tbb::concurrent_hash_map<ConceptId, OntologyConcept>::accessor accessor;
        concept_cache_.insert(accessor, concept.id);
        accessor->second = concept;
    }
    
    return Result<ConceptId>(concept.id);
}

Result<std::vector<ConceptId>> OntologyGraph::createConceptsBatch(
    std::vector<OntologyConcept> concepts) {
    
    std::vector<ConceptId> ids;
    ids.reserve(concepts.size());
    
    rocksdb::WriteBatch batch;
    
    for (auto& concept : concepts) {
        if (concept.id.empty()) {
            concept.id = generateUUID();
        }
        concept.created_at = MemoryTrace::now();
        concept.updated_at = concept.created_at;
        
        std::string data = serializeConcept(concept);
        batch.Put(cf_concepts_, concept.id, data);
        
        // 更新名称索引
        updateNameIndex(concept.id, "", concept.name);
        
        ids.push_back(concept.id);
    }
    
    rocksdb::WriteOptions write_options;
    auto status = db_>Write(write_options, &batch);
    
    if (!status.ok()) {
        return Result<std::vector<ConceptId>>(ErrorCode::STORAGE_WRITE_ERROR,
            std::format("Batch write failed: {}", status.ToString()));
    }
    
    return Result<std::vector<ConceptId>>(std::move(ids));
}

Result<OntologyConcept> OntologyGraph::getConcept(const ConceptId& id) const {
    // 先查缓存
    {
        tbb::concurrent_hash_map<ConceptId, OntologyConcept>::const_accessor accessor;
        if (concept_cache_.find(accessor, id)) {
            return Result<OntologyConcept>(accessor->second);
        }
    }
    
    // 查数据库
    rocksdb::ReadOptions read_options;
    std::string data;
    
    auto status = db_>Get(read_options, cf_concepts_, id, &data);
    
    if (status.IsNotFound()) {
        return Result<OntologyConcept>(ErrorCode::STORAGE_NOT_FOUND,
            std::format("Concept not found: {}", id));
    }
    
    if (!status.ok()) {
        return Result<OntologyConcept>(ErrorCode::STORAGE_READ_ERROR,
            std::format("Failed to read concept: {}", status.ToString()));
    }
    
    auto result = deserializeConcept(data);
    if (result.isError()) {
        return result;
    }
    
    // 添加到缓存
    {
        tbb::concurrent_hash_map<ConceptId, OntologyConcept>::accessor accessor;
        if (concept_cache_.insert(accessor, id)) {
            accessor->second = result.value();
        }
    }
    
    return result;
}

Result<std::vector<OntologyConcept>> OntologyGraph::getConceptsBatch(
    const std::vector<ConceptId>& ids) const {
    
    std::vector<OntologyConcept> concepts;
    concepts.reserve(ids.size());
    
    for (const auto& id : ids) {
        auto result = getConcept(id);
        if (result.isOk()) {
            concepts.push_back(std::move(result.value()));
        }
    }
    
    return Result<std::vector<OntologyConcept>>(std::move(concepts));
}

Result<bool> OntologyGraph::updateConcept(
    const ConceptId& id, const ConceptUpdateRequest& update) {
    
    auto concept_result = getConcept(id);
    if (concept_result.isError()) {
        return Result<bool>(concept_result.errorCode(), concept_result.errorMessage());
    }
    
    auto concept = concept_result.value();
    std::string old_name = concept.name;
    
    // 应用更新
    if (update.name.has_value()) {
        concept.name = update.name.value();
    }
    if (update.aliases.has_value()) {
        concept.aliases = update.aliases.value();
    }
    if (update.concept_type.has_value()) {
        concept.concept_type = update.concept_type.value();
    }
    if (update.description.has_value()) {
        concept.description = update.description.value();
    }
    if (update.properties.has_value()) {
        // 合并属性
        for (const auto& [key, value] : update.properties.value()) {
            concept.properties[key] = value;
        }
    }
    if (update.add_sources.has_value()) {
        concept.sources.insert(concept.sources.end(), 
            update.add_sources.value().begin(), 
            update.add_sources.value().end());
    }
    
    concept.updated_at = MemoryTrace::now();
    
    // 存储更新
    std::string data = serializeConcept(concept);
    rocksdb::WriteOptions write_options;
    auto status = db_>Put(write_options, cf_concepts_, id, data);
    
    if (!status.ok()) {
        return Result<bool>(ErrorCode::STORAGE_WRITE_ERROR,
            std::format("Failed to update concept: {}", status.ToString()));
    }
    
    // 更新索引
    if (concept.name != old_name) {
        updateNameIndex(id, old_name, concept.name);
    }
    
    // 更新缓存
    {
        tbb::concurrent_hash_map<ConceptId, OntologyConcept>::accessor accessor;
        if (concept_cache_.find(accessor, id)) {
            accessor->second = concept;
        }
    }
    
    return Result<bool>(true);
}

Result<bool> OntologyGraph::deleteConcept(const ConceptId& id) {
    // 获取概念以检查是否存在
    auto concept_result = getConcept(id);
    if (concept_result.isError()) {
        return Result<bool>(concept_result.errorCode(), concept_result.errorMessage());
    }
    
    const auto& concept = concept_result.value();
    
    // 删除所有相关关系
    rocksdb::WriteBatch batch;
    
    // 删除出边关系
    for (const auto& rel : concept.outgoing_relations) {
        std::string rel_key = std::format("{}:{}", id, rel.target_concept_id);
        batch.Delete(cf_relations_, rel_key);
    }
    
    // 删除入边关系 (需要扫描)
    rocksdb::ReadOptions read_options;
    std::unique_ptr<rocksdb::Iterator> it(db_>NewIterator(read_options, cf_relations_));
    
    for (it->SeekToFirst(); it->Valid(); it->Next()) {
        auto rel_result = deserializeRelation(it->value().ToString());
        if (rel_result.isOk()) {
            const auto& rel = rel_result.value();
            // 解析key: from_id:to_id
            std::string key = it->key().ToString();
            size_t pos = key.find(':');
            if (pos != std::string::npos) {
                std::string from_id = key.substr(0, pos);
                std::string to_id = key.substr(pos + 1);
                if (to_id == id) {
                    batch.Delete(cf_relations_, it->key());
                }
            }
        }
    }
    
    // 删除概念本身
    batch.Delete(cf_concepts_, id);
    
    rocksdb::WriteOptions write_options;
    auto status = db_>Write(write_options, &batch);
    
    if (!status.ok()) {
        return Result<bool>(ErrorCode::STORAGE_WRITE_ERROR,
            std::format("Failed to delete concept: {}", status.ToString()));
    }
    
    // 更新索引
    updateNameIndex(id, concept.name, "");
    
    // 从缓存移除
    concept_cache_.erase(id);
    
    return Result<bool>(true);
}

bool OntologyGraph::conceptExists(const ConceptId& id) const {
    rocksdb::ReadOptions read_options;
    std::string data;
    auto status = db_>Get(read_options, cf_concepts_, id, &data);
    return status.ok();
}

// =============================================================================
// 关系操作
// =============================================================================

Result<bool> OntologyGraph::addRelation(
    const ConceptId& from, const ConceptRelation& relation) {
    
    // 检查源概念是否存在
    if (!conceptExists(from)) {
        return Result<bool>(ErrorCode::STORAGE_NOT_FOUND,
            std::format("Source concept not found: {}", from));
    }
    
    // 检查目标概念是否存在
    if (!conceptExists(relation.target_concept_id)) {
        return Result<bool>(ErrorCode::STORAGE_NOT_FOUND,
            std::format("Target concept not found: {}", relation.target_concept_id));
    }
    
    // 检查是否已存在相同关系
    std::string rel_key = std::format("{}:{}", from, relation.target_concept_id);
    
    rocksdb::ReadOptions read_options;
    std::string existing;
    auto status = db_>Get(read_options, cf_relations_, rel_key, &existing);
    
    if (status.ok()) {
        return Result<bool>(ErrorCode::STORAGE_ALREADY_EXISTS,
            "Relation already exists");
    }
    
    // 存储关系
    ConceptRelation rel = relation;
    rel.created_at = MemoryTrace::now();
    
    std::string data = serializeRelation(rel);
    rocksdb::WriteOptions write_options;
    status = db_>Put(write_options, cf_relations_, rel_key, data);
    
    if (!status.ok()) {
        return Result<bool>(ErrorCode::STORAGE_WRITE_ERROR,
            std::format("Failed to store relation: {}", status.ToString()));
    }
    
    // 更新概念的出边列表
    auto from_concept_result = getConcept(from);
    if (from_concept_result.isOk()) {
        auto concept = from_concept_result.value();
        concept.outgoing_relations.push_back(rel);
        
        // 更新目标概念的入边列表
        auto to_concept_result = getConcept(relation.target_concept_id);
        if (to_concept_result.isOk()) {
            auto to_concept = to_concept_result.value();
            to_concept.incoming_relations.push_back(rel);
            
            // 批量更新
            rocksdb::WriteBatch batch;
            batch.Put(cf_concepts_, from, serializeConcept(concept));
            batch.Put(cf_concepts_, relation.target_concept_id, serializeConcept(to_concept));
            db_>Write(write_options, &batch);
            
            // 更新缓存
            {
                tbb::concurrent_hash_map<ConceptId, OntologyConcept>::accessor accessor;
                if (concept_cache_.find(accessor, from)) {
                    accessor->second = concept;
                }
                if (concept_cache_.find(accessor, relation.target_concept_id)) {
                    accessor->second = to_concept;
                }
            }
        }
    }
    
    return Result<bool>(true);
}

Result<bool> OntologyGraph::addRelationsBatch(
    const std::vector<std::pair<ConceptId, ConceptRelation>>& relations) {
    
    rocksdb::WriteBatch batch;
    
    for (const auto& [from, relation] : relations) {
        std::string rel_key = std::format("{}:{}", from, relation.target_concept_id);
        std::string data = serializeRelation(relation);
        batch.Put(cf_relations_, rel_key, data);
    }
    
    rocksdb::WriteOptions write_options;
    auto status = db_>Write(write_options, &batch);
    
    if (!status.ok()) {
        return Result<bool>(ErrorCode::STORAGE_WRITE_ERROR,
            std::format("Batch relation write failed: {}", status.ToString()));
    }
    
    return Result<bool>(true);
}

Result<bool> OntologyGraph::removeRelation(
    const ConceptId& from, const ConceptId& to, RelationType type) {
    
    std::string rel_key = std::format("{}:{}", from, to);
    
    rocksdb::WriteOptions write_options;
    auto status = db_>Delete(write_options, cf_relations_, rel_key);
    
    if (!status.ok() && !status.IsNotFound()) {
        return Result<bool>(ErrorCode::STORAGE_WRITE_ERROR,
            std::format("Failed to remove relation: {}", status.ToString()));
    }
    
    // 更新概念的关系列表
    auto from_result = getConcept(from);
    if (from_result.isOk()) {
        auto concept = from_result.value();
        concept.outgoing_relations.erase(
            std::remove_if(concept.outgoing_relations.begin(), 
                          concept.outgoing_relations.end(),
                          [&to, type](const ConceptRelation& r) {
                              return r.target_concept_id == to && r.type == type;
                          }),
            concept.outgoing_relations.end()
        );
        
        rocksdb::WriteOptions write_options;
        db_>Put(write_options, cf_concepts_, from, serializeConcept(concept));
    }
    
    return Result<bool>(true);
}

Result<std::vector<ConceptRelation>> OntologyGraph::queryRelations(
    const ConceptId& concept_id, const RelationQuery& query) const {
    
    auto concept_result = getConcept(concept_id);
    if (concept_result.isError()) {
        return Result<std::vector<ConceptRelation>>(
            concept_result.errorCode(), concept_result.errorMessage());
    }
    
    const auto& concept = concept_result.value();
    std::vector<ConceptRelation> results;
    
    // 出边
    if (query.direction == RelationQuery::Direction::OUTGOING ||
        query.direction == RelationQuery::Direction::BOTH) {
        for (const auto& rel : concept.outgoing_relations) {
            if (matchesRelationQuery(rel, query)) {
                results.push_back(rel);
            }
        }
    }
    
    // 入边
    if (query.direction == RelationQuery::Direction::INCOMING ||
        query.direction == RelationQuery::Direction::BOTH) {
        for (const auto& rel : concept.incoming_relations) {
            // 反转关系用于查询
            if (matchesRelationQuery(rel, query)) {
                results.push_back(rel);
            }
        }
    }
    
    return Result<std::vector<ConceptRelation>>(std::move(results));
}

bool OntologyGraph::matchesRelationQuery(
    const ConceptRelation& rel, const RelationQuery& query) const {
    
    if (query.type_filter.has_value() && rel.type != query.type_filter.value()) {
        return false;
    }
    
    if (query.target_filter.has_value() && 
        rel.target_concept_id != query.target_filter.value()) {
        return false;
    }
    
    if (query.min_confidence.has_value() && 
        rel.confidence < query.min_confidence.value()) {
        return false;
    }
    
    return true;
}

Result<std::vector<ConceptRelation>> OntologyGraph::getRelationsBetween(
    const ConceptId& from, const ConceptId& to) const {
    
    std::vector<ConceptRelation> results;
    
    // 查询从from到to的关系
    std::string rel_key = std::format("{}:{}", from, to);
    rocksdb::ReadOptions read_options;
    std::string data;
    
    auto status = db_>Get(read_options, cf_relations_, rel_key, &data);
    if (status.ok()) {
        auto result = deserializeRelation(data);
        if (result.isOk()) {
            results.push_back(result.value());
        }
    }
    
    // 查询反向关系
    std::string reverse_key = std::format("{}:{}", to, from);
    status = db_>Get(read_options, cf_relations_, reverse_key, &data);
    if (status.ok()) {
        auto result = deserializeRelation(data);
        if (result.isOk()) {
            results.push_back(result.value());
        }
    }
    
    return Result<std::vector<ConceptRelation>>(std::move(results));
}
