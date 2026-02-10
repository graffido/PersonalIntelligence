/**
 * @file hierarchical_memory.h
 * @brief 四层分层记忆存储系统 (v2.0)
 * 
 * 实现认知科学启发的四层存储架构：
 * 
 * ┌─────────────────────────────────────────────────────────────┐
 * │  Layer 0: Sensory Memory (感知记忆)                          │
 * │  - 毫秒级生命周期 (~200-500ms)                               │
 * │  - 原始输入缓冲：视觉、听觉、文本预处理                        │
 * │  - 自动衰减，选择性注意进入工作记忆                            │
 * └─────────────────────────────────────────────────────────────┘
 *                              ↓ 注意机制
 * ┌─────────────────────────────────────────────────────────────┐
 * │  Layer 1: Working Memory (工作记忆)                          │
 * │  - 秒级生命周期 (~30-60s)                                    │
 * │  - 当前任务上下文，注意力焦点                                  │
 * │  - 有限容量 (7±2 个信息块)                                    │
 * │  - 纯内存存储，超高速访问                                      │
 * └─────────────────────────────────────────────────────────────┘
 *                              ↓ 编码巩固
 * ┌─────────────────────────────────────────────────────────────┐
 * │  Layer 2: Long-term Memory (长期记忆)                        │
 * │  - 持久存储，理论上无上限                                       │
 * │  - 显式记忆：情景记忆 + 语义记忆                                │
 * │  - 向量索引 (HNSW) + 时空索引 + 图关系索引                     │
 * │  - 磁盘存储 (RocksDB) + 压缩                                   │
 * └─────────────────────────────────────────────────────────────┘
 *                              ↓ 统计学习
 * ┌─────────────────────────────────────────────────────────────┐
 * │  Layer 3: Parameter Memory (参数记忆)                        │
 * │  - 隐式记忆：程序性知识、统计模式                               │
 * │  - 模型权重、访问模式、用户偏好                                  │
 * │  - 神经网络风格存储，渐进式更新                                  │
 * │  - 支持在线学习和自适应调整                                     │
 * └─────────────────────────────────────────────────────────────┘
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
#include <array>
#include <atomic>
#include <chrono>
#include <thread>
#include <condition_variable>

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

// ============================================================================
// 记忆层类型枚举
// ============================================================================

enum class MemoryLayer : uint8_t {
    SENSORY = 0,        // 感知记忆层
    WORKING = 1,        // 工作记忆层  
    LONG_TERM = 2,      // 长期记忆层
    PARAMETER = 3       // 参数记忆层
};

// ============================================================================
// 各层配置
// ============================================================================

/**
 * @brief 感知记忆配置
 */
struct SensoryMemoryConfig {
    size_t buffer_size = 100;               // 缓冲区大小
    uint32_t decay_ms = 500;                // 衰减时间 (毫秒)
    uint32_t attention_threshold = 60;      // 注意力阈值 (0-100)
    bool enable_multimodal = true;          // 启用多模态处理
};

/**
 * @brief 工作记忆配置
 */
struct WorkingMemoryConfig {
    size_t capacity = 7;                    // 容量 (7±2 规则)
    uint32_t max_age_ms = 60000;            // 最大保留时间 (60秒)
    uint32_t rehearsal_interval_ms = 10000; // 复述间隔
    bool enable_chunking = true;            // 启用信息分块
};

/**
 * @brief 长期记忆配置
 */
struct LongTermMemoryConfig {
    std::string storage_path = "./data/ltm";
    size_t lru_cache_size = 10000;          // LRU缓存大小
    size_t vector_dim = 768;                // 向量维度
    size_t hnsw_m = 16;                     // HNSW M参数
    size_t hnsw_ef_construction = 200;      // HNSW ef_construction
    size_t hnsw_ef_search = 50;             // HNSW ef_search
    std::string hnsw_index_path = "./data/hnsw_index";
    size_t rtree_max_elements = 10000;      // RTree最大元素数
    bool enable_compression = true;         // 启用压缩
};

/**
 * @brief 参数记忆配置
 */
struct ParameterMemoryConfig {
    std::string storage_path = "./data/param_memory";
    size_t embedding_size = 256;            // 嵌入维度
    float learning_rate = 0.001f;           // 学习率
    float decay_factor = 0.99f;             // 衰减因子
    uint32_t update_interval_ms = 300000;   // 更新间隔 (5分钟)
    bool enable_online_learning = true;     // 启用在线学习
};

/**
 * @brief 完整存储配置
 */
struct HierarchicalMemoryConfig {
    SensoryMemoryConfig sensory;
    WorkingMemoryConfig working;
    LongTermMemoryConfig long_term;
    ParameterMemoryConfig parameter;
    
    // 全局设置
    bool enable_auto_consolidation = true;  // 自动巩固
    bool enable_forgetting = true;          // 启用遗忘机制
    uint32_t maintenance_interval_ms = 60000; // 维护任务间隔
};

// ============================================================================
// 感知记忆结构
// ============================================================================

/**
 * @brief 感官输入类型
 */
enum class SensoryType : uint8_t {
    TEXT = 0,
    IMAGE = 1,
    AUDIO = 2,
    VIDEO = 3,
    SENSOR = 4,     // 传感器数据
    CUSTOM = 5
};

/**
 * @brief 感知记忆条目
 * 原始输入的临时缓冲
 */
struct SensoryBufferEntry {
    MemoryId id;
    SensoryType type;
    std::vector<uint8_t> raw_data;          // 原始数据
    Timestamp timestamp;
    float attention_weight;                 // 注意力权重 (0-100)
    std::optional<Embedding> embedding;     // 预计算嵌入
    std::optional<GeoPoint> location;       // 位置信息
    
    explicit SensoryBufferEntry(SensoryType t = SensoryType::TEXT)
        : id(generateUUID())
        , type(t)
        , timestamp(MemoryTrace::now())
        , attention_weight(0.0f) {}
    
    // 检查是否应进入工作记忆
    bool deservesAttention() const {
        return attention_weight >= 60.0f;
    }
    
    // 检查是否已衰减
    bool isExpired(Timestamp now, uint32_t decay_ms) const {
        return (now - timestamp) > decay_ms;
    }
};

// ============================================================================
// 工作记忆结构
// ============================================================================

/**
 * @brief 信息块 (Chunk)
 * 工作记忆中的信息聚合单元
 */
struct MemoryChunk {
    MemoryId id;
    std::string label;                      // 块标签
    std::vector<MemoryId> items;            // 包含的记忆ID
    Timestamp timestamp;
    uint32_t access_count;
    float importance;
    
    MemoryChunk() : id(generateUUID()), timestamp(MemoryTrace::now()), 
                    access_count(0), importance(0.5f) {}
};

/**
 * @brief 工作记忆条目
 */
struct WorkingMemoryEntry {
    MemoryTrace memory;
    Timestamp timestamp;
    uint32_t rehearsal_count;               // 复述计数
    float activation_level;                 // 激活水平 (0-1)
    bool is_focused;                        // 是否处于注意力焦点
    std::optional<MemoryChunk> chunk;       // 所属信息块
    
    explicit WorkingMemoryEntry(MemoryTrace m)
        : memory(std::move(m))
        , timestamp(MemoryTrace::now())
        , rehearsal_count(0)
        , activation_level(1.0f)
        , is_focused(false) {}
    
    // 计算衰减后的激活水平
    float computeActivation(Timestamp now, uint32_t decay_ms) const {
        float time_decay = std::exp(-static_cast<float>(now - timestamp) / decay_ms);
        float rehearsal_boost = std::min(1.0f, rehearsal_count * 0.1f);
        return std::min(1.0f, activation_level * time_decay + rehearsal_boost);
    }
};

// ============================================================================
// 长期记忆结构
// ============================================================================

/**
 * @brief 记忆访问统计
 */
struct AccessStatistics {
    uint64_t access_count = 0;
    Timestamp last_access = 0;
    Timestamp first_access = 0;
    double importance_score = 0.5;
    std::vector<Timestamp> access_history;  // 最近访问历史
    
    void recordAccess() {
        if (first_access == 0) first_access = MemoryTrace::now();
        last_access = MemoryTrace::now();
        access_count++;
        access_history.push_back(last_access);
        // 只保留最近100次访问
        if (access_history.size() > 100) {
            access_history.erase(access_history.begin());
        }
        updateImportance();
    }
    
    void updateImportance() {
        auto now = MemoryTrace::now();
        double recency = (now - last_access) < 86400000 ? 1.0 : 
                        1.0 / (1.0 + (now - last_access) / 86400000.0);
        double frequency = std::min(1.0, access_count / 100.0);
        double pattern_strength = computeAccessPatternStrength();
        importance_score = 0.3 * recency + 0.4 * frequency + 0.3 * pattern_strength;
    }
    
    double computeAccessPatternStrength() const {
        if (access_history.size() < 2) return 0.5;
        // 计算访问间隔的规律性
        double variance = 0;
        double mean_interval = 0;
        for (size_t i = 1; i < access_history.size(); i++) {
            mean_interval += (access_history[i] - access_history[i-1]);
        }
        mean_interval /= (access_history.size() - 1);
        for (size_t i = 1; i < access_history.size(); i++) {
            double diff = (access_history[i] - access_history[i-1]) - mean_interval;
            variance += diff * diff;
        }
        variance /= (access_history.size() - 1);
        // 规律性越强，分数越高
        return std::exp(-variance / 1000000.0);
    }
};

/**
 * @brief 长期记忆条目
 */
struct LongTermMemoryEntry {
    MemoryTrace memory;
    AccessStatistics stats;
    Timestamp consolidated_at;              // 巩固时间
    std::vector<std::string> tags;          // 语义标签
    float emotional_valence;                // 情感效价 (-1 到 1)
    
    explicit LongTermMemoryEntry(MemoryTrace m)
        : memory(std::move(m))
        , consolidated_at(MemoryTrace::now())
        , emotional_valence(0.0f) {}
};

// ============================================================================
// 参数记忆结构
// ============================================================================

/**
 * @brief 用户偏好向量
 */
struct UserPreferenceVector {
    std::array<float, 256> embedding;       // 固定维度偏好嵌入
    std::unordered_map<std::string, float> scalar_prefs; // 标量偏好
    Timestamp last_updated;
    uint32_t update_count;
    
    UserPreferenceVector() : last_updated(MemoryTrace::now()), update_count(0) {
        embedding.fill(0.0f);
    }
};

/**
 * @brief 访问模式模型
 */
struct AccessPatternModel {
    std::vector<float> temporal_weights;    // 时间访问权重 (24小时分布)
    std::vector<float> category_weights;    // 类别访问权重
    std::vector<float> query_pattern;       // 查询模式嵌入
    float entropy;                          // 访问熵 (随机性度量)
    
    AccessPatternModel() : entropy(1.0f) {
        temporal_weights.resize(24, 1.0f/24.0f);
    }
};

/**
 * @brief 参数记忆条目
 */
struct ParameterMemoryEntry {
    MemoryId id;
    std::string key;
    std::vector<float> values;              // 参数向量
    float confidence;                       // 置信度
    Timestamp created_at;
    Timestamp updated_at;
    uint32_t version;                       // 版本号
    
    ParameterMemoryEntry(const std::string& k, const std::vector<float>& v)
        : id(generateUUID())
        , key(k)
        , values(v)
        , confidence(0.5f)
        , created_at(MemoryTrace::now())
        , updated_at(created_at)
        , version(1) {}
};

// ============================================================================
// 四层记忆存储类
// ============================================================================

class QuadLayerMemoryStore {
public:
    explicit QuadLayerMemoryStore(const HierarchicalMemoryConfig& config);
    ~QuadLayerMemoryStore();
    
    // 禁用拷贝，允许移动
    QuadLayerMemoryStore(const QuadLayerMemoryStore&) = delete;
    QuadLayerMemoryStore& operator=(const QuadLayerMemoryStore&) = delete;
    QuadLayerMemoryStore(QuadLayerMemoryStore&&) noexcept;
    QuadLayerMemoryStore& operator=(QuadLayerMemoryStore&&) noexcept;
    
    // 初始化和关闭
    [[nodiscard]] Result<bool> initialize();
    void shutdown();
    [[nodiscard]] bool isInitialized() const noexcept { return initialized_; }
    
    // ========================================================================
    // Layer 0: 感知记忆操作
    // ========================================================================
    
    /**
     * @brief 接收感官输入
     */
    [[nodiscard]] Result<MemoryId> sensoryInput(
        SensoryType type,
        const std::vector<uint8_t>& raw_data,
        float attention_hint = 0.0f);
    
    /**
     * @brief 处理文本输入
     */
    [[nodiscard]] Result<MemoryId> sensoryInputText(
        const std::string& text,
        float attention_hint = 0.0f);
    
    /**
     * @brief 获取感知记忆中的注意力焦点
     */
    [[nodiscard]] std::vector<SensoryBufferEntry> getAttentionFocus();
    
    /**
     * @brief 手动触发注意机制
     */
    void triggerAttention(const std::vector<MemoryId>& ids);
    
    // ========================================================================
    // Layer 1: 工作记忆操作
    // ========================================================================
    
    /**
     * @brief 将记忆加载到工作记忆
     */
    [[nodiscard]] Result<bool> loadToWorkingMemory(
        const MemoryTrace& memory,
        bool set_focus = false);
    
    /**
     * @brief 获取当前焦点记忆
     */
    [[nodiscard]] std::optional<MemoryTrace> getFocusedMemory();
    
    /**
     * @brief 设置注意力焦点
     */
    void setFocus(const MemoryId& id);
    
    /**
     * @brief 复述记忆 (增强激活)
     */
    [[nodiscard]] Result<bool> rehearse(const MemoryId& id);
    
    /**
     * @brief 获取工作记忆中的所有条目
     */
    [[nodiscard]] std::vector<WorkingMemoryEntry> getWorkingMemoryContents();
    
    /**
     * @brief 创建工作记忆信息块
     */
    [[nodiscard]] Result<MemoryId> createChunk(
        const std::string& label,
        const std::vector<MemoryId>& items);
    
    // ========================================================================
    // Layer 2: 长期记忆操作
    // ========================================================================
    
    /**
     * @brief 存储到长期记忆
     */
    [[nodiscard]] Result<MemoryId> storeLongTerm(const MemoryTrace& memory);
    
    /**
     * @brief 从长期记忆检索
     */
    [[nodiscard]] Result<MemoryTrace> retrieveLongTerm(const MemoryId& id);
    
    /**
     * @brief 语义搜索
     */
    [[nodiscard]] Result<std::vector<ScoredItem<MemoryTrace>>> semanticSearch(
        const Embedding& embedding,
        size_t top_k = 10,
        double min_score = 0.7);
    
    /**
     * @brief 时空查询
     */
    [[nodiscard]] Result<std::vector<MemoryTrace>> spatiotemporalQuery(
        std::optional<Timestamp> time_start,
        std::optional<Timestamp> time_end,
        std::optional<GeoBoundingBox> bbox,
        size_t limit = 100);
    
    /**
     * @brief 关联两个记忆
     */
    [[nodiscard]] Result<bool> associate(
        const MemoryId& id1,
        const MemoryId& id2,
        double strength = 1.0);
    
    // ========================================================================
    // Layer 3: 参数记忆操作
    // ========================================================================
    
    /**
     * @brief 获取用户偏好
     */
    [[nodiscard]] UserPreferenceVector getUserPreferences();
    
    /**
     * @brief 更新用户偏好
     */
    void updateUserPreferences(
        const std::vector<float>& gradient,
        float learning_rate = 0.001f);
    
    /**
     * @brief 获取访问模式
     */
    [[nodiscard]] AccessPatternModel getAccessPattern();
    
    /**
     * @brief 记录访问事件 (用于模式学习)
     */
    void recordAccessEvent(const MemoryId& id, MemoryLayer layer);
    
    /**
     * @brief 存储参数
     */
    [[nodiscard]] Result<MemoryId> storeParameter(
        const std::string& key,
        const std::vector<float>& values);
    
    /**
     * @brief 检索参数
     */
    [[nodiscard]] std::optional<std::vector<float>> retrieveParameter(
        const std::string& key);
    
    // ========================================================================
    // 层间转换
    // ========================================================================
    
    /**
     * @brief 感知 → 工作记忆 (通过注意机制)
     */
    [[nodiscard]] Result<MemoryId> attend(
        const MemoryId& sensory_id,
        const std::string& interpreted_content);
    
    /**
     * @brief 工作 → 长期记忆 (巩固)
     */
    [[nodiscard]] Result<MemoryId> consolidate(
        const MemoryId& working_id);
    
    /**
     * @brief 长期 → 工作记忆 (回忆)
     */
    [[nodiscard]] Result<bool> recallToWorking(const MemoryId& ltm_id);
    
    /**
     * @brief 长期记忆 → 参数记忆 (学习)
     */
    [[nodiscard]] Result<bool> learnFromExperiences();
    
    // ========================================================================
    // 记忆管理
    // ========================================================================
    
    /**
     * @brief 触发完整记忆维护
     */
    [[nodiscard]] Result<MemoryMaintenanceResult> runMaintenance();
    
    /**
     * @brief 获取各层统计
     */
    struct LayerStatistics {
        size_t sensory_count = 0;
        size_t working_count = 0;
        size_t working_chunks = 0;
        size_t long_term_count = 0;
        size_t parameter_count = 0;
        size_t vector_index_count = 0;
        size_t spatial_index_count = 0;
        double avg_access_latency_ms = 0.0;
        double cache_hit_rate = 0.0;
    };
    [[nodiscard]] LayerStatistics getStatistics() const;
    
    /**
     * @brief 创建检查点
     */
    [[nodiscard]] Result<bool> createCheckpoint(const std::string& path);
    
    /**
     * @brief 从检查点恢复
     */
    [[nodiscard]] Result<bool> restoreFromCheckpoint(const std::string& path);

private:
    HierarchicalMemoryConfig config_;
    std::atomic<bool> initialized_{false};
    
    // ========================================================================
    // Layer 0: 感知记忆 - 环形缓冲区
    // ========================================================================
    mutable std::shared_mutex sensory_mutex_;
    std::deque<SensoryBufferEntry> sensory_buffer_;
    std::atomic<size_t> sensory_head_{0};
    std::atomic<size_t> sensory_tail_{0};
    
    // 注意机制
    std::unordered_set<MemoryId> attention_focus_;
    std::mutex attention_mutex_;
    
    // ========================================================================
    // Layer 1: 工作记忆
    // ========================================================================
    mutable std::shared_mutex working_mutex_;
    std::unordered_map<MemoryId, WorkingMemoryEntry> working_memory_;
    std::optional<MemoryId> focused_id_;
    std::unordered_map<MemoryId, MemoryChunk> chunks_;
    
    // ========================================================================
    // Layer 2: 长期记忆
    // ========================================================================
    mutable std::shared_mutex ltm_mutex_;
    std::unique_ptr<rocksdb::DB> ltm_db_;
    rocksdb::ColumnFamilyHandle* ltm_cf_memories_ = nullptr;
    rocksdb::ColumnFamilyHandle* ltm_cf_embeddings_ = nullptr;
    rocksdb::ColumnFamilyHandle* ltm_cf_indices_ = nullptr;
    rocksdb::ColumnFamilyHandle* ltm_cf_relations_ = nullptr;
    
    // LRU缓存
    std::unordered_map<MemoryId, LongTermMemoryEntry> ltm_cache_;
    std::list<MemoryId> ltm_lru_list_;
    
    // HNSW向量索引
    std::unique_ptr<hnswlib::L2Space> hnsw_space_;
    std::unique_ptr<hnswlib::HierarchicalNSW<float>> hnsw_index_;
    mutable std::shared_mutex hnsw_mutex_;
    
    // RTree空间索引
    using RTree = bgi::rtree<SpatialValue, bgi::rstar<16>>;
    std::unique_ptr<RTree> spatial_index_;
    mutable std::shared_mutex spatial_mutex_;
    
    // 时间索引
    std::map<Timestamp, std::set<MemoryId>> temporal_index_;
    mutable std::shared_mutex temporal_mutex_;
    
    // ========================================================================
    // Layer 3: 参数记忆
    // ========================================================================
    mutable std::shared_mutex param_mutex_;
    std::unique_ptr<rocksdb::DB> param_db_;
    
    UserPreferenceVector user_prefs_;
    AccessPatternModel access_pattern_;
    std::unordered_map<std::string, ParameterMemoryEntry> parameters_;
    
    // 访问统计 (用于学习)
    std::deque<std::pair<Timestamp, MemoryId>> access_history_;
    std::array<std::atomic<uint32_t>, 24> hourly_access_count_;
    
    // ========================================================================
    // 统计信息
    // ========================================================================
    mutable std::atomic<size_t> cache_hits_{0};
    mutable std::atomic<size_t> cache_misses_{0};
    mutable std::atomic<double> total_access_time_ms_{0.0};
    mutable std::atomic<size_t> access_count_{0};
    
    // ========================================================================
    // 后台维护
    // ========================================================================
    std::unique_ptr<std::thread> maintenance_thread_;
    std::atomic<bool> stop_maintenance_{false};
    std::condition_variable maintenance_cv_;
    std::mutex maintenance_mutex_;
    
    // ========================================================================
    // 内部方法
    // ========================================================================
    
    // 初始化各层
    [[nodiscard]] Result<bool> initializeSensoryLayer();
    [[nodiscard]] Result<bool> initializeWorkingLayer();
    [[nodiscard]] Result<bool> initializeLongTermLayer();
    [[nodiscard]] Result<bool> initializeParameterLayer();
    
    // 维护任务
    void maintenanceLoop();
    void decaySensoryMemory();
    void decayWorkingMemory();
    void consolidateWorkingToLTM();
    void updateAccessPatternModel();
    void cleanupExpiredRelations();
    
    // 索引管理
    [[nodiscard]] Result<bool> addToHnswIndex(
        const MemoryId& id, const Embedding& embedding);
    [[nodiscard]] Result<bool> removeFromHnswIndex(const MemoryId& id);
    [[nodiscard]] std::vector<std::pair<MemoryId, double>> searchHnsw(
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
    [[nodiscard]] std::string serializeStats(const AccessStatistics& stats) const;
    [[nodiscard]] Result<AccessStatistics> deserializeStats(const std::string& data) const;
    
    // 缓存管理
    void addToLtmCache(const MemoryId& id, const LongTermMemoryEntry& entry);
    void removeFromLtmCache(const MemoryId& id);
    [[nodiscard]] std::optional<LongTermMemoryEntry> getFromLtmCache(const MemoryId& id);
    
    // 学习算法
    void updateTemporalWeights();
    void updateCategoryWeights();
    [[nodiscard]] float computeEntropy(const std::vector<float>& distribution);
};

/**
 * @brief 维护结果
 */
struct MemoryMaintenanceResult {
    size_t decayed_sensory = 0;
    size_t decayed_working = 0;
    size_t consolidated = 0;
    size_t forgotten = 0;
    size_t learned_params = 0;
    double duration_ms = 0.0;
};

} // namespace memory
} // namespace personal_ontology
