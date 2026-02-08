#include "reasoning_engine.h"
#include <algorithm>
#include <set>
#include <math>
#include <iostream>

namespace pos {

// 构造函数
LocalReasoningEngine::LocalReasoningEngine(
    OntologyGraph& ontology,
    MemoryStore& memory,
    SpatiotemporalIndex& st_index
) : ontology_(ontology), memory_(memory), st_index_(st_index) {
    registerBuiltinRules();
}

LocalReasoningEngine::~LocalReasoningEngine() = default;

// 注册内置规则
void LocalReasoningEngine::registerBuiltinRules() {
    // 1. 社交传递性规则
    rules_.push_back(ReasoningRuleFactory::friendOfFriendRule());
    
    // 2. 日程冲突检测
    rules_.push_back(ReasoningRuleFactory::scheduleConflictRule());
    
    // 3. 地点可达性
    rules_.push_back(ReasoningRuleFactory::locationReachabilityRule());
    
    // 4. 习惯模式识别
    rules_.push_back(ReasoningRuleFactory::habitPatternRule());
    
    // 5. 情感一致性
    rules_.push_back(ReasoningRuleFactory::emotionalCoherenceRule());
    
    // 6. 重复事件
    rules_.push_back(ReasoningRuleFactory::recurringEventRule());
    
    // 7. 社交关系强度
    rules_.push_back(ReasoningRuleFactory::relationshipStrengthRule());
}

// 注册规则
void LocalReasoningEngine::registerRule(const InferenceRule& rule) {
    rules_.push_back(rule);
}

void LocalReasoningEngine::unregisterRule(const std::string& rule_id) {
    rules_.erase(
        std::remove_if(rules_.begin(), rules_.end(),
            [&rule_id](const auto& r) { return r.id == rule_id; }),
        rules_.end()
    );
}

// 主推理函数
std::vector<InferenceResult> LocalReasoningEngine::infer(
    const std::optional<ConceptId>& focus_concept,
    const std::optional<Timestamp>& focus_time
) {
    std::vector<InferenceResult> results;
    
    // 1. 关系传递性推理
    if (focus_concept) {
        auto transitive = inferTransitiveRelations(*focus_concept);
        results.insert(results.end(), transitive.begin(), transitive.end());
    }
    
    // 2. 冲突检测
    if (focus_time) {
        auto window_start = *focus_time - std::chrono::hours(24);
        auto window_end = *focus_time + std::chrono::hours(24);
        auto conflicts = detectConflicts(window_start, window_end);
        results.insert(results.end(), conflicts.begin(), conflicts.end());
    }
    
    // 3. 模式发现
    auto patterns = discoverPatterns("all", 2);
    for (const auto& pattern : patterns) {
        InferenceResult result;
        result.rule_id = "pattern_discovery";
        result.rule_name = "模式发现";
        result.type = "pattern";
        result.description = pattern.description;
        result.confidence = pattern.confidence;
        result.supporting_memories = pattern.examples;
        results.push_back(result);
    }
    
    // 4. 应用自定义规则
    for (const auto& rule : rules_) {
        if (!rule.enabled) continue;
        
        if (rule.condition(ontology_, memory_)) {
            // 执行规则动作
            rule.action(ontology_, memory_);
            
            InferenceResult result;
            result.rule_id = rule.id;
            result.rule_name = rule.name;
            result.type = "inference";
            result.description = rule.description;
            result.confidence = rule.confidence;
            results.push_back(result);
        }
    }
    
    return results;
}

// 传递性关系推理
std::vector<InferenceResult> LocalReasoningEngine::inferTransitiveRelations(
    const ConceptId& concept
) {
    std::vector<InferenceResult> results;
    
    auto concept_opt = ontology_.getConcept(concept);
    if (!concept_opt) return results;
    
    auto c = *concept_opt;
    
    // 对于每种关系类型，查找传递闭包
    // 例如: KNOWS关系传递 - 朋友的朋友
    for (const auto& rel : c.relations) {
        if (rel.type == RelationType::KNOWS) {
            // 获取目标概念
            auto target_opt = ontology_.getConcept(rel.target);
            if (!target_opt) continue;
            
            auto target = *target_opt;
            
            // 查找目标的KNOWS关系
            for (const auto& target_rel : target.relations) {
                if (target_rel.type == RelationType::KNOWS &&
                    target_rel.target != concept) {  // 不包括回指
                    
                    // 发现传递关系: concept -> target_rel.target
                    InferenceResult result;
                    result.rule_id = "transitive_knows";
                    result.rule_name = "社交关系传递性";
                    result.type = "new_relation";
                    result.description = c.label + " 可能认识 " + 
                        ontology_.getConcept(target_rel.target).value().label +
                        " (通过 " + target.label + " 介绍)";
                    result.confidence = rel.weight * target_rel.weight * 0.8f;
                    result.involved_concepts = {concept, target_rel.target};
                    result.suggested_action = "建议引荐认识";
                    results.push_back(result);
                }
            }
        }
    }
    
    return results;
}

// 冲突检测
std::vector<InferenceResult> LocalReasoningEngine::detectConflicts(
    const Timestamp& time_window_start,
    const Timestamp& time_window_end
) {
    std::vector<InferenceResult> results;
    
    // 获取时间窗口内的所有记忆
    auto memories = memory_.retrieveByTime(time_window_start, time_window_end, 100);
    
    // 两两比较检查冲突
    for (size_t i = 0; i < memories.size(); ++i) {
        for (size_t j = i + 1; j < memories.size(); ++j) {
            const auto& mem1 = memories[i].memory;
            const auto& mem2 = memories[j].memory;
            
            // 检查时间重叠
            bool time_overlap = false;
            if (mem1.end_time && mem2.timestamp < *mem1.end_time) {
                time_overlap = true;
            }
            if (mem2.end_time && mem1.timestamp < *mem2.end_time) {
                time_overlap = true;
            }
            
            if (!time_overlap) continue;
            
            // 检查是否需要同时出现在两个地点
            if (mem1.location && mem2.location) {
                double distance = mem1.location->distanceTo(*mem2.location);
                
                // 如果在不同地点且时间重叠，可能是冲突
                if (distance > 1000) {  // 1km以上
                    InferenceResult result;
                    result.rule_id = "schedule_conflict";
                    result.rule_name = "日程冲突检测";
                    result.type = "conflict";
                    result.description = "时间冲突: " + mem1.content.raw_text + 
                        " vs " + mem2.content.raw_text;
                    result.confidence = 0.9f;
                    result.supporting_memories = {mem1.id, mem2.id};
                    
                    InferenceResult::Conflict conflict;
                    conflict.conflict_type = "location_conflict";
                    conflict.severity = "high";
                    conflict.resolution_suggestion = "需要重新安排其中一个日程";
                    result.conflict = conflict;
                    
                    results.push_back(result);
                }
            }
        }
    }
    
    return results;
}

// 生成推荐
std::vector<InferenceResult> LocalReasoningEngine::generateRecommendations(
    const SituationContext& current_situation
) {
    std::vector<InferenceResult> results;
    
    // 基于当前情境和历史模式生成推荐
    auto patterns = discoverPatterns("behavioral", 3);
    
    for (const auto& pattern : patterns) {
        if (pattern.prediction && pattern.prediction->probability > 0.6f) {
            InferenceResult result;
            result.rule_id = "pattern_based_recommendation";
            result.rule_name = "基于模式的推荐";
            result.type = "suggestion";
            result.description = "根据您的习惯，建议: " + pattern.prediction->predicted_event;
            result.confidence = pattern.prediction->probability;
            result.suggested_action = pattern.prediction->predicted_event;
            results.push_back(result);
        }
    }
    
    // 基于社交关系的推荐
    // 例如: 如果A和B是朋友，A喜欢某餐厅，推荐给B
    
    return results;
}

// 模式发现
std::vector<Pattern> LocalReasoningEngine::discoverPatterns(
    const std::string& pattern_type,
    int min_frequency
) {
    std::vector<Pattern> patterns;
    
    // 1. 时间模式发现
    if (pattern_type == "all" || pattern_type == "temporal") {
        // 获取所有记忆
        auto all_memories = memory_.retrieveByTime(
            std::chrono::system_clock::now() - std::chrono::days(90),
            std::chrono::system_clock::now(),
            1000
        );
        
        // 按概念分组
        std::map<ConceptId, std::vector<MemoryId>> concept_memories;
        for (const auto& result : all_memories) {
            for (const auto& concept : result.memory.ontology_bindings) {
                concept_memories[concept].push_back(result.memory.id);
            }
        }
        
        // 查找重复模式
        for (const auto& [concept_id, memory_ids] : concept_memories) {
            if (memory_ids.size() >= static_cast<size_t>(min_frequency)) {
                auto pattern = discoverTemporalPattern(memory_ids);
                if (pattern.frequency >= min_frequency) {
                    auto concept_opt = ontology_.getConcept(concept_id);
                    if (concept_opt) {
                        pattern.description = "经常与 " + concept_opt->label + " 相关的活动";
                        patterns.push_back(pattern);
                    }
                }
            }
        }
    }
    
    return patterns;
}

// 发现时间模式
Pattern LocalReasoningEngine::discoverTemporalPattern(
    const std::vector<MemoryId>& memories
) {
    Pattern pattern;
    pattern.pattern_type = "temporal";
    pattern.examples = memories;
    pattern.frequency = memories.size();
    
    // 分析时间分布
    std::map<std::string, int> time_of_day_count;
    std::map<int, int> day_of_week_count;
    
    for (const auto& mem_id : memories) {
        auto mem_opt = memory_.getById(mem_id);
        if (!mem_opt) continue;
        
        auto time_of_day = getTimeOfDay(mem_opt->timestamp);
        time_of_day_count[time_of_day]++;
        
        std::time_t time = std::chrono::system_clock::to_time_t(mem_opt->timestamp);
        std::tm tm = *std::localtime(&time);
        day_of_week_count[tm.tm_wday]++;
    }
    
    // 找出最常见的时间段
    std::string dominant_time;
    int max_count = 0;
    for (const auto& [time, count] : time_of_day_count) {
        if (count > max_count) {
            max_count = count;
            dominant_time = time;
        }
    }
    
    if (!dominant_time.empty()) {
        TemporalPattern tp;
        tp.time_of_day = dominant_time;
        tp.recurrence_type = "custom";
        pattern.temporal_info = tp;
        pattern.confidence = static_cast<float>(max_count) / memories.size();
    }
    
    // 生成预测
    if (pattern.confidence > 0.5f) {
        Pattern::Prediction pred;
        pred.predicted_event = "下次可能在" + dominant_time + "进行类似活动";
        pred.probability = pattern.confidence;
        
        // 预测下次时间 (简化: 假设按平均间隔)
        pred.predicted_time = std::chrono::system_clock::now() + std::chrono::days(7);
        
        pattern.prediction = pred;
    }
    
    return pattern;
}

// 预测下一个事件
std::vector<InferenceResult> LocalReasoningEngine::predictNextEvents(
    const Timestamp& from_time,
    int prediction_horizon_hours
) {
    std::vector<InferenceResult> results;
    
    // 基于发现的模式进行预测
    auto patterns = discoverPatterns("all", 2);
    
    for (const auto& pattern : patterns) {
        if (pattern.prediction) {
            InferenceResult result;
            result.rule_id = "pattern_prediction";
            result.rule_name = "模式预测";
            result.type = "suggestion";
            result.description = pattern.prediction->predicted_event;
            result.confidence = pattern.prediction->probability;
            results.push_back(result);
        }
    }
    
    return results;
}

// 规则工厂实现
InferenceRule ReasoningRuleFactory::friendOfFriendRule() {
    InferenceRule rule;
    rule.id = "friend_of_friend";
    rule.name = "朋友的朋友";
    rule.description = "朋友的朋友可能是潜在联系人";
    rule.type = RuleType::TRANSITIVE;
    rule.confidence = 0.7f;
    
    rule.condition = [](const OntologyGraph& onto, const MemoryStore& mem) {
        // 检查是否有足够多的社交关系
        auto persons = onto.findConceptsByType(ConceptType::PERSON, 100);
        return persons.size() >= 3;
    };
    
    rule.action = [](OntologyGraph& onto, MemoryStore& mem) {
        // 实际推理在inferTransitiveRelations中完成
    };
    
    return rule;
}

InferenceRule ReasoningRuleFactory::scheduleConflictRule() {
    InferenceRule rule;
    rule.id = "schedule_conflict";
    rule.name = "日程冲突检测";
    rule.description = "检测时间和空间上的冲突";
    rule.type = RuleType::TEMPORAL;
    rule.confidence = 0.9f;
    
    rule.condition = [](const OntologyGraph& onto, const MemoryStore& mem) {
        // 检查是否有待处理的日程
        return mem.getMemoryCount() > 0;
    };
    
    rule.action = [](OntologyGraph& onto, MemoryStore& mem) {
        // 冲突检测在detectConflicts中完成
    };
    
    return rule;
}

InferenceRule ReasoningRuleFactory::locationReachabilityRule() {
    InferenceRule rule;
    rule.id = "location_reachable";
    rule.name = "地点可达性";
    rule.description = "检查是否能在时间限制内到达目的地";
    rule.type = RuleType::SPATIAL;
    rule.confidence = 0.85f;
    
    return rule;
}

InferenceRule ReasoningRuleFactory::habitPatternRule() {
    InferenceRule rule;
    rule.id = "habit_pattern";
    rule.name = "习惯模式识别";
    rule.description = "从重复行为中发现习惯模式";
    rule.type = RuleType::CUSTOM;
    rule.confidence = 0.75f;
    
    return rule;
}

InferenceRule ReasoningRuleFactory::emotionalCoherenceRule() {
    InferenceRule rule;
    rule.id = "emotional_coherence";
    rule.name = "情感一致性";
    rule.description = "检测情绪状态与活动的一致性";
    rule.type = RuleType::CUSTOM;
    rule.confidence = 0.6f;
    
    return rule;
}

InferenceRule ReasoningRuleFactory::recurringEventRule() {
    InferenceRule rule;
    rule.id = "recurring_event";
    rule.name = "重复事件检测";
    rule.description = "发现周期性发生的事件";
    rule.type = RuleType::TEMPORAL;
    rule.confidence = 0.8f;
    
    return rule;
}

InferenceRule ReasoningRuleFactory::relationshipStrengthRule() {
    InferenceRule rule;
    rule.id = "relationship_strength";
    rule.name = "关系强度计算";
    rule.description = "基于互动频率计算关系强度";
    rule.type = RuleType::CUSTOM;
    rule.confidence = 0.7f;
    
    return rule;
}

} // namespace pos
