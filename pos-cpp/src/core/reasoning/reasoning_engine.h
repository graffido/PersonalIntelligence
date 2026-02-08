#pragma once

#include "core/ontology/ontology_graph.h"
#include "core/memory/memory_store.h"
#include "core/temporal/spatiotemporal_engine.h"
#include <functional>
#include <rule>

namespace pos {

// 推理规则类型
enum class RuleType {
    TRANSITIVE,      // 传递性: A->B, B->C => A->C
    SYMMETRIC,       // 对称性: A->B => B->A
    INVERSE,         // 反向: A->B => B->A的反向关系
    COMPOSITION,     // 组合: A->B + B->C => A->C
    DOMAIN_RANGE,    // 定义域/值域约束
    CARDINALITY,     // 基数约束
    TEMPORAL,        // 时间约束
    SPATIAL,         // 空间约束
    CUSTOM           // 自定义规则
};

// 推理规则定义
struct InferenceRule {
    std::string id;
    std::string name;
    std::string description;
    RuleType type;
    float confidence{1.0};
    bool enabled{true};
    
    // 规则条件 (lambda)
    std::function<bool(const OntologyGraph&, const MemoryStore&)> condition;
    
    // 规则动作 (lambda)
    std::function<void(OntologyGraph&, MemoryStore&)> action;
};

// 推理结果
struct InferenceResult {
    std::string rule_id;
    std::string rule_name;
    std::string type;           // "new_relation", "conflict", "suggestion", "pattern"
    std::string description;
    float confidence{0.0};
    std::vector<ConceptId> involved_concepts;
    std::vector<MemoryId> supporting_memories;
    std::optional<std::string> suggested_action;
    
    // 对于冲突检测
    struct Conflict {
        std::string conflict_type;
        std::string severity;   // "low", "medium", "high"
        std::string resolution_suggestion;
    };
    std::optional<Conflict> conflict;
};

// 模式发现结果
struct Pattern {
    std::string pattern_type;   // "temporal", "spatial", "social", "behavioral"
    std::string description;
    float confidence{0.0};
    int frequency{0};           // 出现次数
    std::vector<MemoryId> examples;
    std::optional<TemporalPattern> temporal_info;
    std::optional<GeoPoint> spatial_center;
    
    // 预测信息
    struct Prediction {
        std::string predicted_event;
        float probability{0.0};
        std::optional<Timestamp> predicted_time;
        std::optional<GeoPoint> predicted_location;
    };
    std::optional<Prediction> prediction;
};

// 时间模式
struct TemporalPattern {
    std::string recurrence_type;  // "daily", "weekly", "monthly", "custom"
    std::string time_of_day;      // "morning", "afternoon", "evening", "night"
    std::vector<int> days_of_week; // 0=Sunday, 1=Monday, ...
    int interval_days{0};         // 间隔天数
    std::optional<Timestamp> next_occurrence;
};

// 本地推理引擎
class LocalReasoningEngine {
public:
    explicit LocalReasoningEngine(
        OntologyGraph& ontology,
        MemoryStore& memory,
        SpatiotemporalIndex& st_index
    );
    
    ~LocalReasoningEngine();
    
    // 注册推理规则
    void registerRule(const InferenceRule& rule);
    void unregisterRule(const std::string& rule_id);
    
    // 执行推理
    std::vector<InferenceResult> infer(
        const std::optional<ConceptId>& focus_concept = std::nullopt,
        const std::optional<Timestamp>& focus_time = std::nullopt
    );
    
    // 特定推理类型
    std::vector<InferenceResult> inferTransitiveRelations(const ConceptId& concept);
    std::vector<InferenceResult> detectConflicts(const Timestamp& time_window_start,
                                                    const Timestamp& time_window_end);
    std::vector<InferenceResult> generateRecommendations(
        const SituationContext& current_situation
    );
    
    // 模式发现
    std::vector<Pattern> discoverPatterns(
        const std::string& pattern_type = "all",
        int min_frequency = 2
    );
    
    // 预测
    std::vector<InferenceResult> predictNextEvents(
        const Timestamp& from_time,
        int prediction_horizon_hours = 24
    );
    
    // 验证
    bool validateConsistency();
    std::vector<InferenceResult> findInconsistencies();

private:
    OntologyGraph& ontology_;
    MemoryStore& memory_;
    SpatiotemporalIndex& st_index_;
    
    std::vector<InferenceRule> rules_;
    
    // 内置规则实现
    void registerBuiltinRules();
    
    // 具体推理实现
    InferenceResult applyTransitiveRule(const ConceptId& a, const ConceptId& b, 
                                        const ConceptId& c, RelationType rel_type);
    
    InferenceResult detectScheduleConflict(const MemoryId& mem1, const MemoryId& mem2);
    
    InferenceResult detectLocationUnreachable(
        const MemoryId& from_mem, 
        const MemoryId& to_mem,
        const std::string& transport_mode
    );
    
    Pattern discoverTemporalPattern(const std::vector<MemoryId>& memories);
    
    // 辅助函数
    float calculateRelationConfidence(const ConceptId& from, 
                                     const ConceptId& to, 
                                     RelationType type);
    
    bool isRuleApplicable(const InferenceRule& rule);
};

// 推理规则工厂 - 预定义常用规则
class ReasoningRuleFactory {
public:
    // 社交关系传递性: 朋友的朋友可能是潜在联系人
    static InferenceRule friendOfFriendRule();
    
    // 日程冲突检测
    static InferenceRule scheduleConflictRule();
    
    // 地点可达性检查
    static InferenceRule locationReachabilityRule();
    
    // 习惯模式识别
    static InferenceRule habitPatternRule();
    
    // 情感一致性检查
    static InferenceRule emotionalCoherenceRule();
    
    // 重复事件检测
    static InferenceRule recurringEventRule();
    
    // 社交关系强度计算
    static InferenceRule relationshipStrengthRule();
    
    // 时间冲突预警
    static InferenceRule timeConflictWarningRule();
};

} // namespace pos
