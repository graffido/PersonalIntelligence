/**
 * @file memory_store.h
 * @brief 分层记忆存储系统
 * 
 * 实现三层存储架构：
 * - 工作记忆 (Working Memory): 高频访问，纯内存
 * - 短期记忆 (Short-term Memory): 最近访问，内存+磁盘
 * - 长期记忆 (Long-term Memory): 完整历史，磁盘存储
 * 
 * 集成HNSW向量索引和时空索引
 */

#pragma once

#include "../common/types.h"
#include <memory>
#include <queue>
#include <mutex>
#include <shared_mutex>
#include <rocksdb/db.h>
#include <boost/geometry.hpp>
#include <boost/geometry/index/rtree.hpp>

// HNSW库前向声明
namespace hnswlib {
    template<typename dist_t>
    class HierarchicalNSW;
    class L2Space;
}

namespace personal_ontology {
namespace memory {

// 空间索引使用Boost.Geometry
namespace bg = boost::geometry;
namespace bgi = boost::geometry::index;

// 空间索引点类型
using Point = bg::model::point<double, 2, bg::cs::cartesian>;
using SpatialBox = bg::model::box<Point>;
using SpatialValue = std::pair<Point, MemoryId>;

/**
 * @brief 存储层配置
 */
struct StorageLayerConfig {
    // 工作记忆
    size_t working_memory_capacity = 1000;    // 最大条目数
    size_t working_memory_max_age_ms = 60000; // 最大保留时间(1分钟)
    
    // 短期记忆
    size_t short_term_capacity = 10000;       // 最大条目数
    size_t short_term_batch_size = 100;       // 批处理大小
    std::string short_term_path = "/tmp/personal_ontology_stm";
    
    // 长期记忆
    std::string long_term_path = "./data/memories";
    size_t lru_cache_size = 10000;            // LRU缓存大小
    
    // 向量索引
    size_t vector_dim = 768;                  // 向量维度
    size_t hnsw_m = 16;                       // HNSW M参数
    size_t hnsw_ef_construction = 200;        // HNSW ef_construction
    size_t hnsw_ef_search = 50;               // HNSW ef_search
    std::string hnsw_index_path = "./data/hnsw_index";
    
    // 时空索引
    size_t rtree_max_elements = 10000;        // RTree最大元素数
};

/**
 * @brief 记忆访问统计
 */
struct AccessStats {
    uint64_t access_count = 0;
    Timestamp last_access = 0;
    Timestamp first_access = 0;
    double importance_score = 0.5;
    
    void recordAccess() {
        if (first_access == 0) first_access = MemoryTrace::now();
        last_access = MemoryTrace::now();
        access_count++;
        // 重要性衰减公式
        updateImportance();
    }
    
    void updateImportance() {
        // 基于访问频率和最近访问时间的重要性计算
        auto now = MemoryTrace::now();
        double recency = (now - last_access) < 86400000 ? 1.0 : 
                        1.0 / (1.0 + (now - last_access) / 86400000.0);
        double frequency = std::min(1.0, access_count / 100.0);
        importance_score = 0.4 * recency + 0.6 * frequency;
    }
};

/**
 * @brief 记忆条目包装器
 */
struct MemoryEntry {
    MemoryTrace memory;
    AccessStats stats;
    uint8_t current_layer;  // 当前存储层
    
    explicit MemoryEntry(MemoryTrace m, uint8_t layer = 0) 
        : memory(std::move(m)), current_layer(layer) {
        stats.first_access = MemoryTrace::now();
    }
};

/**
 * @brief 分层记忆存储类
 */
class HierarchicalMemoryStore {
public:
    /**
     * @brief 构造函数
     * @param config 存储配置
     */
    explicit HierarchicalMemoryStore(const StorageLayerConfig& config);
    
    /**
     * @brief 析构函数
     */
    ~HierarchicalMemoryStore();
    
    // 禁用拷贝，允许移动
    HierarchicalMemoryStore(const HierarchicalMemoryStore&) = delete;
    HierarchicalMemoryStore& operator=(const HierarchicalMemoryStore&) = delete;
    HierarchicalMemoryStore(HierarchicalMemoryStore&&) noexcept;
    HierarchicalMemoryStore& operator=(HierarchicalMemoryStore&&) noexcept;
    
    /**
     * @brief 初始化存储系统
     * @return 是否成功
     */
    [[nodiscard]] Result<bool> initialize();
    
    /**
     * @brief 关闭存储系统
     */
    void shutdown();
    
    /**
     * @brief 检查是否已初始化
     */
    [[nodiscard]] bool isInitialized() const noexcept { return initialized_; }
    
    // ========================================================================
    // 核心CRUD操作
    // ========================================================================
    
    /**
     * @brief 存储记忆
     * @param memory 记忆数据
     * @param target_layer 目标存储层 (0=工作记忆, 1=短期, 2=长期)
     * @return 记忆ID或错误
     */
    [[nodiscard]] Result<MemoryId> store(
        MemoryTrace memory, 
        uint8_t target_layer = 0);
    
    /**
     * @brief 批量存储记忆
     * @param memories 记忆列表
     * @param target_layer 目标存储层
     * @return ID列表或错误
     */
    [[nodiscard]] Result<std::vector<MemoryId>> storeBatch(
        std::vector<MemoryTrace> memories,
        uint8_t target_layer = 1);
    
    /**
     * @brief 获取记忆
     * @param id 记忆ID
     * @return 记忆或错误
     */
    [[nodiscard]] Result<MemoryTrace> retrieve(const MemoryId& id);
    
    /**
     * @brief 批量获取记忆
     * @param ids ID列表
     * @return 记忆列表或错误
     */
    [[nodiscard]] Result<std::vector<MemoryTrace>> retrieveBatch(
        const std::vector<MemoryId>& ids);
    
    /**
     * @brief 更新记忆
     * @param id 记忆ID
     * @param update 更新函数
     * @return 是否成功
     */
    [[nodiscard]] Result<bool> update(
        const MemoryId& id,
        std::function<void(MemoryTrace&)> update);
    
    /**
     * @brief 删除记忆
     * @param id 记忆ID
     * @return 是否成功
     */
    [[nodiscard]] Result<bool> remove(const MemoryId& id);
    
    /**
     * @brief 批量删除记忆
     * @param ids ID列表
     * @return 删除数量或错误
     */
    [[nodiscard]] Result<size_t> removeBatch(const std::vector<MemoryId>& ids);
    
    // ========================================================================
    // 查询操作
    // ========================================================================
    
    /**
     * @brief 执行记忆查询
     * @param query 查询条件
     * @return 查询结果或错误
     */
    [[nodiscard]] Result<MemoryQueryResult> query(const MemoryQuery& query);
    
    /**
     * @brief 语义相似度搜索
     * @param embedding 查询向量
     * @param top_k 返回数量
     * @param min_score 最小相似度
     * @return 带分数的记忆列表或错误
     */
    [[nodiscard]] Result<std::vector<ScoredItem<MemoryTrace>>> semanticSearch(
        const Embedding& embedding,
        size_t top_k = 10,
        double min_score = 0.7);
    
    /**
     * @brief 多模态搜索 (向量 + 元数据过滤)
     * @param embedding 查询向量
     * @param query 元数据过滤条件
     * @param top_k 返回数量
     * @return 带分数的记忆列表或错误
     */
    [[nodiscard]] Result<std::vector<ScoredItem<MemoryTrace>>> multimodalSearch(
        const Embedding& embedding,
        const MemoryQuery& query,
        size_t top_k = 10);
    
    /**
     * @brief 时空范围查询
     * @param time_start 开始时间
     * @param time_end 结束时间
     * @param bbox 空间范围
     * @param limit 返回数量
     * @return 记忆列表或错误
     */
    [[nodiscard]] Result<std::vector<MemoryTrace>> spatiotemporalQuery(
        std::optional<Timestamp> time_start,
        std::optional<Timestamp> time_end,
        std::optional<GeoBoundingBox> bbox,
        size_t limit = 100);
    
    /**
     * @brief 获取实体相关记忆
     * @param entity_id 实体ID
     * @param limit 返回数量
     * @return 记忆列表或错误
     */
    [[nodiscard]] Result<std::vector<MemoryTrace>> getEntityMemories(
        const EntityId& entity_id,
        size_t limit = 100);
    
    /**
     * @brief 获取概念相关记忆
     * @param concept_id 概念ID
     * @param limit 返回数量
     * @return 记忆列表或错误
     */
    [[nodiscard]] Result<std::vector<MemoryTrace>> getConceptMemories(
        const ConceptId& concept_id,
        size_t limit = 100);
    
    // ========================================================================
    // 记忆管理
    // ========================================================================
    
    /**
     * @brief 触发记忆巩固
     * 将重要记忆从短期迁移到长期
     * @return 处理数量或错误
     */
    [[nodiscard]] Result<size_t> consolidate();
    
    /**
     * @brief 触发记忆遗忘
     * 移除不重要或过期的记忆
     * @return 删除数量或错误
     */
    [[nodiscard]] Result<size_t> forget();
    
    /**
     * @brief 清理过期记忆
     * @return 删除数量或错误
     */
    [[nodiscard]] Result<size_t> cleanupExpired();
    
    /**
     * @brief 提升记忆重要性
     * @param id 记忆ID
     * @param boost_factor 提升因子
     * @return 是否成功
     */
    [[nodiscard]] Result<bool> boostImportance(
        const MemoryId& id, 
        double boost_factor = 1.5);
    
    /**
     * @brief 关联两个记忆
     * @param id1 记忆1 ID
     * @param id2 记忆2 ID
     * @param strength 关联强度
     * @return 是否成功
     */
    [[nodiscard]] Result<bool> associate(
        const MemoryId& id1,
        const MemoryId& id2,
        double strength = 1.0);
    
    // ========================================================================
    // 统计和监控
    // ========================================================================
    
    /**
     * @brief 获取存储统计
     */
    struct Statistics {
        size_t working_memory_count = 0;
        size_t short_term_count = 0;
        size_t long_term_count = 0;
        size_t vector_index_count = 0;
        size_t spatial_index_count = 0;
        double avg_access_latency_ms = 0.0;
        size_t cache_hit_count = 0;
        size_t cache_miss_count = 0;
    };
    [[nodiscard]] Statistics getStatistics() const;
    
    /**
     * @brief 获取记忆访问统计
     * @param id 记忆ID
     * @return 访问统计或错误
     */
    [[nodiscard]] Result<AccessStats> getAccessStats(const MemoryId& id) const;
    
    /**
     * @brief 创建检查点
     * @param checkpoint_path 检查点路径
     * @return 是否成功
     */
    [[nodiscard]] Result<bool> createCheckpoint(
        const std::string& checkpoint_path);
    
private:
    StorageLayerConfig config_;
    
    // 初始化状态
    std::atomic<bool> initialized_{false};
    
    // 工作记忆 - 纯内存，线程安全
    mutable std::shared_mutex wm_mutex_;
    std::unordered_map<MemoryId, MemoryEntry> working_memory_;
    std::queue<MemoryId> wm_lru_queue_;
    
    // 短期记忆 - 内存缓存 + RocksDB
    mutable std::shared_mutex stm_mutex_;
    std::unordered_map<MemoryId, MemoryEntry> short_term_cache_;
    std::unique_ptr<rocksdb::DB> stm_db_;
    rocksdb::ColumnFamilyHandle* stm_cf_default_ = nullptr;
    rocksdb::ColumnFamilyHandle* stm_cf_metadata_ = nullptr;
    
    // 长期记忆 - RocksDB
    mutable std::shared_mutex ltm_mutex_;
    std::unique_ptr<rocksdb::DB> ltm_db_;
    rocksdb::ColumnFamilyHandle* ltm_cf_memories_ = nullptr;
    rocksdb::ColumnFamilyHandle* ltm_cf_embeddings_ = nullptr;
    rocksdb::ColumnFamilyHandle* ltm_cf_indices_ = nullptr;
    
    // 向量索引 - HNSW
    std::unique_ptr<hnswlib::L2Space> hnsw_space_;
    std::unique_ptr<hnswlib::HierarchicalNSW<float>> hnsw_index_;
    mutable std::shared_mutex hnsw_mutex_;
    
    // 时空索引 - RTree
    using RTree = bgi::rtree<SpatialValue, bgi::rstar<16>>;
    std::unique_ptr<RTree> spatial_index_;
    mutable std::shared_mutex spatial_mutex_;
    
    // 时间索引 (按时间排序)
    std::map<Timestamp, std::set<MemoryId>> temporal_index_;
    mutable std::shared_mutex temporal_mutex_;
    
    // 统计信息
    mutable std::atomic<size_t> cache_hits_{0};
    mutable std::atomic<size_t> cache_misses_{0};
    mutable std::atomic<double> total_access_time_ms_{0.0};
    mutable std::atomic<size_t> access_count_{0};
    
    // ========================================================================
    // 内部辅助方法
    // ========================================================================
    
    // 层间迁移
    [[nodiscard]] Result<bool> migrateToShortTerm(const MemoryId& id);
    [[nodiscard]] Result<bool> migrateToLongTerm(const MemoryId& id);
    [[nodiscard]] Result<bool> loadIntoWorkingMemory(const MemoryId& id);
    
    // 索引管理
    [[nodiscard]] Result<bool> addToHnswIndex(
        const MemoryId& id, const Embedding& embedding);
    [[nodiscard]] Result<bool> removeFromHnswIndex(const MemoryId& id);
    [[nodiscard]] Result<std::vector<std::pair<MemoryId, double>>> searchHnsw(
        const Embedding& embedding, size_t top_k);
    
    [[nodiscard]] Result<bool> addToSpatialIndex(
        const MemoryId& id, const GeoPoint& location);
    [[nodiscard]] Result<bool> removeFromSpatialIndex(const MemoryId& id);
    [[nodiscard]] std::vector<MemoryId> searchSpatial(
        const GeoBoundingBox& bbox);
    
    [[nodiscard]] Result<bool> addToTemporalIndex(
        const MemoryId& id, Timestamp time);
    [[nodiscard]] Result<bool> removeFromTemporalIndex(const MemoryId& id);
    [[nodiscard]] std::vector<MemoryId> searchTemporal(
        Timestamp start, Timestamp end);
    
    // 序列化
    [[nodiscard]] std::string serializeMemory(const MemoryTrace& memory) const;
    [[nodiscard]] Result<MemoryTrace> deserializeMemory(const std::string& data) const;
    [[nodiscard]] std::string serializeAccessStats(const AccessStats& stats) const;
    [[nodiscard]] Result<AccessStats> deserializeAccessStats(const std::string& data) const;
    
    // 存储层操作
    [[nodiscard]] Result<bool> writeToStm(const MemoryEntry& entry);
    [[nodiscard]] Result<std::optional<MemoryEntry>> readFromStm(const MemoryId& id);
    [[nodiscard]] Result<bool> deleteFromStm(const MemoryId& id);
    
    [[nodiscard]] Result<bool> writeToLtm(const MemoryEntry& entry);
    [[nodiscard]] Result<std::optional<MemoryTrace>> readFromLtm(const MemoryId& id);
    [[nodiscard]] Result<bool> deleteFromLtm(const MemoryId& id);
    
    // 内存管理
    void evictFromWorkingMemory();
    void evictFromShortTermCache();
    
    // 维护任务
    void runMaintenanceTasks();
};

} // namespace memory
} // namespace personal_ontology
