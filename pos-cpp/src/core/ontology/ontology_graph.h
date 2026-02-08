/**
 * @file ontology_graph.h
 * @brief 本体图谱接口定义
 * 
 * 定义了本体图数据结构、操作接口和存储抽象
 * 使用RocksDB作为底层存储
 */

#pragma once

#include "../common/types.h"
#include <memory>
#include <rocksdb/db.h>
#include <rocksdb/options.h>
#include <tbb/concurrent_hash_map.h>

namespace personal_ontology {
namespace ontology {

// 前向声明
class OntologyGraph;

/**
 * @brief 图谱配置选项
 */
struct OntologyGraphConfig {
    std::string data_path;                    // RocksDB数据路径
    size_t cache_size_mb = 256;               // 块缓存大小(MB)
    int max_open_files = 1000;                // 最大打开文件数
    bool enable_compression = true;           // 启用压缩
    bool create_if_missing = true;            // 不存在时创建
    
    // 写入选项
    bool sync_writes = false;                 // 同步写入
    bool wal_enabled = true;                  // 启用预写日志
};

/**
 * @brief 概念更新请求
 */
struct ConceptUpdateRequest {
    std::optional<std::string> name;              // 更新名称
    std::optional<std::vector<std::string>> aliases;  // 更新别名
    std::optional<ConceptType> concept_type;      // 更新类型
    std::optional<std::string> description;       // 更新描述
    std::optional<Properties> properties;         // 更新属性 (合并)
    std::optional<std::vector<std::string>> add_sources; // 添加来源
};

/**
 * @brief 关系查询条件
 */
struct RelationQuery {
    std::optional<RelationType> type_filter;      // 关系类型过滤
    std::optional<ConceptId> target_filter;       // 目标概念过滤
    std::optional<double> min_confidence;         // 最小置信度
    
    enum class Direction {
        OUTGOING,     // 只查出边
        INCOMING,     // 只查入边
        BOTH          // 双向查询
    };
    Direction direction = Direction::OUTGOING;
};

/**
 * @brief 语义相似度查询结果
 */
struct SemanticSimilarityResult {
    ConceptId concept_id;
    std::string concept_name;
    double similarity_score;
    ConceptType type;
};

/**
 * @brief 图谱统计信息
 */
struct GraphStatistics {
    size_t concept_count = 0;                 // 概念数量
    size_t relation_count = 0;                // 关系数量
    size_t type_distribution[9] = {0};        // 各类型概念数量
    
    // 关系类型分布
    std::unordered_map<RelationType, size_t> relation_distribution;
    
    // 存储统计
    size_t db_size_bytes = 0;                 // 数据库大小
    size_t memtable_size = 0;                 // 内存表大小
};

/**
 * @brief 图谱遍历访问器
 */
class GraphTraverser {
public:
    virtual ~GraphTraverser() = default;
    
    // 访问节点时的回调
    virtual bool visitNode(const OntologyConcept& concept, int depth) = 0;
    
    // 访问边时的回调
    virtual bool visitEdge(const ConceptId& from, const ConceptRelation& relation, int depth) = 0;
    
    // 决定是否继续遍历某条路径
    virtual bool shouldTraverse(const ConceptRelation& relation, int current_depth) = 0;
};

/**
 * @brief 本体图谱类
 * 
 * 管理概念的CRUD操作、关系管理和语义查询
 * 线程安全设计，支持并发访问
 */
class OntologyGraph {
public:
    /**
     * @brief 构造函数
     * @param config 图谱配置
     */
    explicit OntologyGraph(const OntologyGraphConfig& config);
    
    /**
     * @brief 析构函数
     */
    ~OntologyGraph();
    
    // 禁用拷贝，允许移动
    OntologyGraph(const OntologyGraph&) = delete;
    OntologyGraph& operator=(const OntologyGraph&) = delete;
    OntologyGraph(OntologyGraph&&) noexcept;
    OntologyGraph& operator=(OntologyGraph&&) noexcept;
    
    /**
     * @brief 初始化图谱
     * @return 是否成功
     */
    [[nodiscard]] Result<bool> initialize();
    
    /**
     * @brief 关闭图谱，释放资源
     */
    void shutdown();
    
    /**
     * @brief 检查是否已初始化
     */
    [[nodiscard]] bool isInitialized() const noexcept { return initialized_; }
    
    // =========================================================================
    // 概念CRUD操作
    // =========================================================================
    
    /**
     * @brief 创建新概念
     * @param concept 概念数据 (id可空, 会自动生成)
     * @return 创建的概念ID或错误
     */
    [[nodiscard]] Result<ConceptId> createConcept(OntologyConcept concept);
    
    /**
     * @brief 批量创建概念
     * @param concepts 概念列表
     * @return 创建的ID列表或错误
     */
    [[nodiscard]] Result<std::vector<ConceptId>> createConceptsBatch(
        std::vector<OntologyConcept> concepts);
    
    /**
     * @brief 根据ID获取概念
     * @param id 概念ID
     * @return 概念或错误
     */
    [[nodiscard]] Result<OntologyConcept> getConcept(const ConceptId& id) const;
    
    /**
     * @brief 批量获取概念
     * @param ids ID列表
     * @return 概念列表或错误
     */
    [[nodiscard]] Result<std::vector<OntologyConcept>> getConceptsBatch(
        const std::vector<ConceptId>& ids) const;
    
    /**
     * @brief 更新概念
     * @param id 概念ID
     * @param update 更新请求
     * @return 是否成功
     */
    [[nodiscard]] Result<bool> updateConcept(
        const ConceptId& id, const ConceptUpdateRequest& update);
    
    /**
     * @brief 删除概念及其所有关系
     * @param id 概念ID
     * @return 是否成功
     */
    [[nodiscard]] Result<bool> deleteConcept(const ConceptId& id);
    
    /**
     * @brief 检查概念是否存在
     * @param id 概念ID
     * @return 是否存在
     */
    [[nodiscard]] bool conceptExists(const ConceptId& id) const;
    
    // =========================================================================
    // 关系操作
    // =========================================================================
    
    /**
     * @brief 添加概念间关系
     * @param from 源概念ID
     * @param relation 关系定义
     * @return 是否成功
     */
    [[nodiscard]] Result<bool> addRelation(
        const ConceptId& from, const ConceptRelation& relation);
    
    /**
     * @brief 批量添加关系
     * @param relations 关系列表 (from_id, relation)
     * @return 是否成功
     */
    [[nodiscard]] Result<bool> addRelationsBatch(
        const std::vector<std::pair<ConceptId, ConceptRelation>>& relations);
    
    /**
     * @brief 移除关系
     * @param from 源概念ID
     * @param to 目标概念ID
     * @param type 关系类型
     * @return 是否成功
     */
    [[nodiscard]] Result<bool> removeRelation(
        const ConceptId& from, const ConceptId& to, RelationType type);
    
    /**
     * @brief 查询概念关系
     * @param concept_id 概念ID
     * @param query 查询条件
     * @return 关系列表或错误
     */
    [[nodiscard]] Result<std::vector<ConceptRelation>> queryRelations(
        const ConceptId& concept_id, const RelationQuery& query) const;
    
    /**
     * @brief 获取两个概念间的关系
     * @param from 源概念ID
     * @param to 目标概念ID
     * @return 关系列表或错误
     */
    [[nodiscard]] Result<std::vector<ConceptRelation>> getRelationsBetween(
        const ConceptId& from, const ConceptId& to) const;
    
    // =========================================================================
    // 查询操作
    // =========================================================================
    
    /**
     * @brief 根据名称查找概念
     * @param name 概念名称
     * @param include_aliases 是否包含别名
     * @return 概念列表或错误
     */
    [[nodiscard]] Result<std::vector<OntologyConcept>> findConceptsByName(
        const std::string& name, bool include_aliases = true) const;
    
    /**
     * @brief 语义相似度搜索
     * @param query_embedding 查询向量
     * @param top_k 返回结果数
     * @param type_filter 类型过滤
     * @return 相似概念列表或错误
     */
    [[nodiscard]] Result<std::vector<SemanticSimilarityResult>> semanticSearch(
        const Embedding& query_embedding, 
        size_t top_k = 10,
        std::optional<ConceptType> type_filter = std::nullopt) const;
    
    /**
     * @brief 全文搜索概念
     * @param keyword 关键词
     * @param limit 返回数量
     * @return 概念列表或错误
     */
    [[nodiscard]] Result<std::vector<OntologyConcept>> searchConcepts(
        const std::string& keyword, size_t limit = 50) const;
    
    /**
     * @brief 获取所有概念ID (分页)
     * @param limit 数量限制
     * @param offset 偏移量
     * @return ID列表或错误
     */
    [[nodiscard]] Result<std::vector<ConceptId>> getAllConceptIds(
        size_t limit = 1000, size_t offset = 0) const;
    
    // =========================================================================
    // 图谱遍历和推理
    // =========================================================================
    
    /**
     * @brief 深度优先遍历
     * @param start_id 起始概念ID
     * @param traverser 遍历访问器
     * @param max_depth 最大深度
     * @return 是否成功
     */
    [[nodiscard]] Result<bool> traverseDepthFirst(
        const ConceptId& start_id,
        GraphTraverser& traverser,
        int max_depth = 5) const;
    
    /**
     * @brief 广度优先遍历
     * @param start_id 起始概念ID
     * @param traverser 遍历访问器
     * @param max_depth 最大深度
     * @return 是否成功
     */
    [[nodiscard]] Result<bool> traverseBreadthFirst(
        const ConceptId& start_id,
        GraphTraverser& traverser,
        int max_depth = 5) const;
    
    /**
     * @brief 查找概念路径
     * @param from 起始概念
     * @param to 目标概念
     * @param max_hops 最大跳数
     * @return 路径 (关系链) 或错误
     */
    [[nodiscard]] Result<std::vector<ConceptRelation>> findPath(
        const ConceptId& from, const ConceptId& to, int max_hops = 5) const;
    
    /**
     * @brief 获取概念的所有子概念 (IS_A关系)
     * @param concept_id 概念ID
     * @param recursive 是否递归获取
     * @return 子概念列表或错误
     */
    [[nodiscard]] Result<std::vector<OntologyConcept>> getSubConcepts(
        const ConceptId& concept_id, bool recursive = false) const;
    
    /**
     * @brief 获取概念的所有父概念 (IS_A关系)
     * @param concept_id 概念ID
     * @param recursive 是否递归获取
     * @return 父概念列表或错误
     */
    [[nodiscard]] Result<std::vector<OntologyConcept>> getSuperConcepts(
        const ConceptId& concept_id, bool recursive = false) const;
    
    /**
     * @brief 计算概念间的语义距离
     * @param id1 概念1
     * @param id2 概念2
     * @return 距离值 (越小越相似) 或错误
     */
    [[nodiscard]] Result<double> calculateSemanticDistance(
        const ConceptId& id1, const ConceptId& id2) const;
    
    // =========================================================================
    // 高级功能
    // =========================================================================
    
    /**
     * @brief 合并两个概念
     * @param keep_id 保留的概念ID
     * @param merge_id 被合并的概念ID
     * @param merge_relations 是否合并关系
     * @return 是否成功
     */
    [[nodiscard]] Result<bool> mergeConcepts(
        const ConceptId& keep_id,
        const ConceptId& merge_id,
        bool merge_relations = true);
    
    /**
     * @brief 分裂概念 (创建新概念并转移部分关系)
     * @param source_id 源概念ID
     * @param new_concept 新概念数据
     * @param relations_to_move 要转移的关系
     * @return 新概念ID或错误
     */
    [[nodiscard]] Result<ConceptId> splitConcept(
        const ConceptId& source_id,
        OntologyConcept new_concept,
        const std::vector<ConceptRelation>& relations_to_move);
    
    /**
     * @brief 导入概念 (从外部数据源)
     * @param concepts 概念列表
     * @param source 数据来源标识
     * @param skip_existing 跳过已存在
     * @return 导入统计或错误
     */
    struct ImportStats {
        size_t created = 0;
        size_t updated = 0;
        size_t skipped = 0;
        size_t failed = 0;
    };
    [[nodiscard]] Result<ImportStats> importConcepts(
        const std::vector<OntologyConcept>& concepts,
        const std::string& source,
        bool skip_existing = true);
    
    /**
     * @brief 导出概念 (用于备份或迁移)
     * @param concept_ids 要导出的ID列表 (空表示全部)
     * @param include_relations 是否包含关系
     * @return 概念数据列表或错误
     */
    [[nodiscard]] Result<std::vector<OntologyConcept>> exportConcepts(
        const std::vector<ConceptId>& concept_ids = {},
        bool include_relations = true) const;
    
    // =========================================================================
    // 维护和统计
    // =========================================================================
    
    /**
     * @brief 获取图谱统计信息
     * @return 统计信息
     */
    [[nodiscard]] GraphStatistics getStatistics() const;
    
    /**
     * @brief 检查并修复数据一致性
     * @param auto_fix 自动修复问题
     * @return 问题列表
     */
    [[nodiscard]] Result<std::vector<std::string>> checkIntegrity(bool auto_fix = false);
    
    /**
     * @brief 压缩数据库
     * @return 是否成功
     */
    [[nodiscard]] Result<bool> compact();
    
    /**
     * @brief 创建检查点
     * @param checkpoint_path 检查点路径
     * @return 是否成功
     */
    [[nodiscard]] Result<bool> createCheckpoint(const std::string& checkpoint_path);
    
private:
    OntologyGraphConfig config_;
    std::unique_ptr<rocksdb::DB> db_;
    rocksdb::ColumnFamilyHandle* cf_concepts_ = nullptr;
    rocksdb::ColumnFamilyHandle* cf_relations_ = nullptr;
    rocksdb::ColumnFamilyHandle* cf_index_ = nullptr;
    bool initialized_ = false;
    
    // 内存缓存 (热点概念)
    mutable tbb::concurrent_hash_map<ConceptId, OntologyConcept> concept_cache_;
    
    // 序列化辅助函数
    [[nodiscard]] std::string serializeConcept(const OntologyConcept& concept) const;
    [[nodiscard]] Result<OntologyConcept> deserializeConcept(const std::string& data) const;
    [[nodiscard]] std::string serializeRelation(const ConceptRelation& relation) const;
    [[nodiscard]] Result<ConceptRelation> deserializeRelation(const std::string& data) const;
    
    // 索引更新
    [[nodiscard]] Result<bool> updateNameIndex(
        const ConceptId& id, const std::string& old_name, const std::string& new_name);
    [[nodiscard]] Result<bool> updateRelationIndices(
        const ConceptId& from, const ConceptRelation& relation, bool add);
    
    // 内部查询辅助
    [[nodiscard]] std::vector<ConceptId> lookupByName(const std::string& name) const;
    
    // HNSW索引用于语义搜索 (可选)
    std::unique_ptr<void, void(*)(void*)> hnsw_index_;
    void buildHnswIndex();
};

} // namespace ontology
} // namespace personal_ontology