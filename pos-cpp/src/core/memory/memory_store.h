#pragma once

#include "core/common/types.h"
#include <mutex>
#include <vector>
#include <queue>

namespace pos {

// 记忆关联结构
struct MemoryAssociation {
    MemoryId target_id;
    std::string type;      // temporal, causal, similar, thematic
    float strength{0.0};   // 0-1
    std::string description;
};

// 记忆痕迹 - 核心数据结构
struct MemoryTrace {
    MemoryId id;
    MemoryType type{MemoryType::EPISODIC};
    ConsolidationStatus status{ConsolidationStatus::RAW};
    
    // 时间属性
    Timestamp timestamp;
    std::optional<Timestamp> end_time;
    std::unordered_map<std::string, std::string> temporal_context; 
    // e.g., {"time_of_day": "evening", "day_type": "weekend", "season": "spring"}
    
    // 空间属性
    std::optional<GeoPoint> location;
    std::optional<std::string> location_name;
    std::optional<std::string> spatial_context;  // e.g., "at_home", "commuting"
    
    // 内容 (多模态)
    struct Content {
        std::string raw_text;
        Embedding embedding;
        std::unordered_map<std::string, std::string> metadata;
        // metadata可包含: image_url, audio_hash, video_frame, sensor_data_json等
    } content;
    
    // 情感标记
    std::vector<EmotionalTag> emotions;
    float overall_valence{0.0};  // 整体情感效价
    
    // 本体绑定 (双向绑定的关键)
    std::unordered_set<ConceptId> ontology_bindings;
    
    // 关联记忆 (记忆图谱)
    std::vector<MemoryAssociation> associations;
    
    // 元数据
    int access_count{0};
    Timestamp last_accessed;
    Timestamp created_at;
    float importance_score{0.5};  // 0-1, 动态计算
    std::string source{"direct"};  // direct, inferred, imported, synthesized
    float confidence{1.0};
    
    // 方法
    void bindToConcept(const ConceptId& concept);
    void addAssociation(const MemoryAssociation& assoc);
    void updateAccess();
    std::string toJson() const;
    static MemoryTrace fromJson(const std::string& json_str);
};

// 记忆检索结果
struct MemoryRetrievalResult {
    MemoryTrace memory;
    float relevance_score{0.0};
    float temporal_score{0.0};
    float spatial_score{0.0};
    float semantic_score{0.0};
    std::string match_type;  // exact, semantic, temporal, spatial, contextual
};

// 记忆存储接口
class MemoryStore {
public:
    virtual ~MemoryStore() = default;
    
    // 存储
    virtual bool store(const MemoryTrace& memory) = 0;
    virtual bool storeBatch(const std::vector<MemoryTrace>& memories) = 0;
    
    // 基础检索
    virtual std::optional<MemoryTrace> getById(const MemoryId& id) const = 0;
    
    // 语义检索 (向量相似度)
    virtual std::vector<MemoryRetrievalResult> retrieveByText(
        const std::string& query,
        const Embedding& query_embedding,
        int limit = 10
    ) = 0;
    
    // 时间检索
    virtual std::vector<MemoryRetrievalResult> retrieveByTime(
        const Timestamp& start,
        const Timestamp& end,
        int limit = 10
    ) = 0;
    
    virtual std::vector<MemoryRetrievalResult> retrieveByTimeOfDay(
        const std::string& time_of_day,  // morning/afternoon/evening/night
        int limit = 10
    ) = 0;
    
    // 空间检索
    virtual std::vector<MemoryRetrievalResult> retrieveByLocation(
        const GeoPoint& center,
        double radius_meters,
        int limit = 10
    ) = 0;
    
    virtual std::vector<MemoryRetrievalResult> retrieveByLocationName(
        const std::string& name,
        int limit = 10
    ) = 0;
    
    // 本体引导检索
    virtual std::vector<MemoryRetrievalResult> retrieveByConcepts(
        const std::vector<ConceptId>& concepts,
        bool require_all = false,  // true=AND, false=OR
        int limit = 10
    ) = 0;
    
    // 综合情境检索
    virtual std::vector<MemoryRetrievalResult> retrieveContextual(
        const std::optional<Timestamp>& time_hint,
        const std::optional<GeoPoint>& location_hint,
        const std::vector<ConceptId>& concept_hints,
        const std::optional<std::string>& text_hint,
        int limit = 10
    ) = 0;
    
    // 记忆链检索 (时间序列)
    virtual std::vector<MemoryTrace> getMemoryChain(
        const MemoryId& start_id,
        int max_depth = 5
    ) = 0;
    
    // 更新
    virtual bool updateAccessStats(const MemoryId& id) = 0;
    virtual bool bindToConcept(const MemoryId& memory, const ConceptId& concept) = 0;
    virtual bool updateAssociations(const MemoryId& memory,
                                   const std::vector<MemoryAssociation>& associations) = 0;
    
    // 遗忘接口
    virtual bool forget(const MemoryId& id, bool permanent = false) = 0;
    virtual size_t cleanupOldMemories(double age_days_threshold) = 0;
    
    // 统计
    virtual size_t getMemoryCount() const = 0;
    virtual size_t getCountByType(MemoryType type) const = 0;
};

// 分层记忆系统实现
class HierarchicalMemoryStore : public MemoryStore {
public:
    explicit HierarchicalMemoryStore(const std::string& data_dir,
                                     size_t vector_dim = 384);
    ~HierarchicalMemoryStore() override;
    
    // MemoryStore接口实现
    bool store(const MemoryTrace& memory) override;
    bool storeBatch(const std::vector<MemoryTrace>& memories) override;
    std::optional<MemoryTrace> getById(const MemoryId& id) const override;
    
    std::vector<MemoryRetrievalResult> retrieveByText(
        const std::string& query,
        const Embedding& query_embedding,
        int limit = 10
    ) override;
    
    std::vector<MemoryRetrievalResult> retrieveByTime(
        const Timestamp& start,
        const Timestamp& end,
        int limit = 10
    ) override;
    
    std::vector<MemoryRetrievalResult> retrieveByTimeOfDay(
        const std::string& time_of_day,
        int limit = 10
    ) override;
    
    std::vector<MemoryRetrievalResult> retrieveByLocation(
        const GeoPoint& center,
        double radius_meters,
        int limit = 10
    ) override;
    
    std::vector<MemoryRetrievalResult> retrieveByLocationName(
        const std::string& name,
        int limit = 10
    ) override;
    
    std::vector<MemoryRetrievalResult> retrieveByConcepts(
        const std::vector<ConceptId>& concepts,
        bool require_all = false,
        int limit = 10
    ) override;
    
    std::vector<MemoryRetrievalResult> retrieveContextual(
        const std::optional<Timestamp>& time_hint,
        const std::optional<GeoPoint>& location_hint,
        const std::vector<ConceptId>& concept_hints,
        const std::optional<std::string>& text_hint,
        int limit = 10
    ) override;
    
    std::vector<MemoryTrace> getMemoryChain(
        const MemoryId& start_id,
        int max_depth = 5
    ) override;
    
    bool updateAccessStats(const MemoryId& id) override;
    bool bindToConcept(const MemoryId& memory, const ConceptId& concept) override;
    bool updateAssociations(const MemoryId& memory,
                           const std::vector<MemoryAssociation>& associations) override;
    
    bool forget(const MemoryId& id, bool permanent = false) override;
    size_t cleanupOldMemories(double age_days_threshold) override;
    
    size_t getMemoryCount() const override;
    size_t getCountByType(MemoryType type) const override;
    
    // 记忆巩固
    void consolidateMemories();
    
    // 模式发现
    std::vector<std::string> discoverTemporalPatterns(int min_occurrences = 3);

private:
    class Impl;
    std::unique_ptr<Impl> pimpl_;
    mutable std::shared_mutex mutex_;
};

} // namespace pos
