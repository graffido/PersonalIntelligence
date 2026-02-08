#pragma once

#include "core/ontology/ontology_graph.h"
#include "core/memory/memory_store.h"
#include "core/temporal/spatiotemporal_engine.h"
#include "core/reasoning/reasoning_engine.h"
#include "ml/ml_service_client.h"
#include <vector>
#include <memory>

namespace pos {

// 统一输入请求
struct UnifiedInputRequest {
    std::string raw_text;
    std::optional<Timestamp> explicit_time;
    std::optional<GeoPoint> explicit_location;
    std::optional<std::string> source;
};

// 实体提取结果
struct ExtractedEntity {
    std::string text;
    std::string label;  // PERSON, PLACE, TIME, EVENT, OBJECT
    size_t start_pos;
    size_t end_pos;
    float confidence;
    std::string normalized_form;
    ConceptId canonical_id;
};

// 关系提取结果
struct ExtractedRelation {
    std::string subject;
    std::string predicate;
    std::string object;
    float confidence;
};

// 统一处理结果
struct UnifiedProcessResult {
    MemoryId memory_id;
    std::vector<ExtractedEntity> entities;
    std::vector<ExtractedRelation> relations;
    std::vector<InferenceResult> reasoning_results;
    
    // 生成的推荐
    struct Recommendation {
        std::string type;
        std::string title;
        std::string description;
        float confidence;
        int priority;
        std::string reason;
    };
    std::vector<Recommendation> recommendations;
    
    // 统计
    int new_concepts_created{0};
    int existing_concepts_linked{0};
};

// 查询请求
struct QueryRequest {
    std::string query_text;
    std::string query_type{"auto"};  // auto, semantic, temporal, spatial, concept
    std::optional<Timestamp> time_hint;
    std::optional<GeoPoint> location_hint;
};

// 查询结果
struct QueryResult {
    std::vector<MemoryTrace> direct_matches;
    std::vector<MemoryTrace> inferred_matches;
    std::vector<InferenceResult> reasoning_path;
    std::string strategy_used;
};

// POS核心引擎 - 系统的核心
class POSCoreEngine {
public:
    // 构造函数
    POSCoreEngine(const std::string& data_dir, 
                  const MLServiceConfig& ml_config);
    
    ~POSCoreEngine();
    
    // 禁止拷贝
    POSCoreEngine(const POSCoreEngine&) = delete;
    POSCoreEngine& operator=(const POSCoreEngine&) = delete;
    
    // ========== 核心API ==========
    
    // 统一输入处理 - 单一入口
    UnifiedProcessResult processInput(const UnifiedInputRequest& request);
    
    // 智能查询
    QueryResult query(const QueryRequest& request);
    
    // 获取推荐
    std::vector<UnifiedProcessResult::Recommendation> 
    generateRecommendations(const SituationContext& context, int limit = 5);
    
    // 预测未来事件
    std::vector<InferenceResult> predictNextEvents(
        const Timestamp& from_time, 
        int horizon_hours = 24
    );
    
    // ========== 管理API ==========
    
    // 获取记忆
    std::optional<MemoryTrace> getMemory(const MemoryId& id);
    
    // 获取概念
    std::optional<OntologyConcept> getConcept(const ConceptId& id);
    
    // 删除记忆
    bool deleteMemory(const MemoryId& id);
    
    // 获取统计
    struct Stats {
        size_t memory_count{0};
        size_t concept_count{0};
        size_t relation_count{0};
        size_t pattern_count{0};
    };
    Stats getStats() const;
    
    // 导出数据
    std::string exportToJson() const;
    
    // 导入数据
    bool importFromJson(const std::string& json_data);
    
    // ========== 推理API ==========
    
    // 手动触发推理
    std::vector<InferenceResult> triggerInference(
        const std::optional<ConceptId>& focus = std::nullopt
    );
    
    // 检测冲突
    std::vector<InferenceResult> detectConflicts(
        const Timestamp& window_start,
        const Timestamp& window_end
    );
    
    // 发现模式
    std::vector<Pattern> discoverPatterns(int min_frequency = 3);

private:
    // 存储组件
    std::unique_ptr<OntologyGraph> ontology_;
    std::unique_ptr<HierarchicalMemoryStore> memory_;
    std::unique_ptr<SpatiotemporalIndex> spatiotemporal_;
    
    // 推理引擎
    std::unique_ptr<LocalReasoningEngine> reasoning_;
    
    // ML客户端 (轻量级)
    std::unique_ptr<MLServiceClient> ml_client_;
    
    // 数据目录
    std::string data_dir_;
    
    // ========== 内部方法 ==========
    
    // 调用ML服务提取实体
    std::vector<ExtractedEntity> extractEntities(const std::string& text);
    
    // 生成embedding
    Embedding generateEmbedding(const std::string& text);
    
    // 解析时间
    Timestamp parseTime(const std::string& text, 
                        const std::optional<Timestamp>& context_time);
    
    // 解析地点
    std::optional<GeoPoint> parseLocation(const std::string& text);
    
    // 创建或获取概念
    ConceptId getOrCreateConcept(const std::string& label, 
                                  ConceptType type,
                                  float confidence);
    
    // 实体消歧
    std::string disambiguateEntity(const std::string& text, 
                                    const std::string& label);
    
    // 生成推荐
    std::vector<UnifiedProcessResult::Recommendation> 
    generateRecommendationsInternal(const SituationContext& context);
};

} // namespace pos
