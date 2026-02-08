#include "pos_core_engine.h"
#include <nlohmann/json.hpp>
#include <chrono>
#include <iostream>

namespace pos {

using json = nlohmann::json;

// 构造函数
POSCoreEngine::POSCoreEngine(const std::string& data_dir,
                               const MLServiceConfig& ml_config)
    : data_dir_(data_dir) {
    
    // 初始化存储
    ontology_ = std::make_unique<OntologyGraph>(data_dir + "/ontology");
    memory_ = std::make_unique<HierarchicalMemoryStore>(data_dir + "/memories", 384);
    spatiotemporal_ = std::make_unique<SpatiotemporalIndex>(data_dir + "/temporal");
    
    // 初始化推理引擎
    reasoning_ = std::make_unique<LocalReasoningEngine>(
        *ontology_, *memory_, *spatiotemporal_
    );
    
    // 初始化ML客户端
    ml_client_ = std::make_unique<MLServiceClient>(ml_config);
    
    std::cout << "[POSCore] Engine initialized" << std::endl;
}

POSCoreEngine::~POSCoreEngine() = default;

// 统一输入处理
UnifiedProcessResult POSCoreEngine::processInput(const UnifiedInputRequest& request) {
    UnifiedProcessResult result;
    
    // 1. 调用ML服务提取实体
    auto entities = extractEntities(request.raw_text);
    result.entities = entities;
    
    // 2. 生成embedding
    auto embedding = generateEmbedding(request.raw_text);
    
    // 3. 解析时间
    auto parsed_time = request.explicit_time.has_value() 
        ? *request.explicit_time 
        : parseTime(request.raw_text, std::nullopt);
    
    // 4. 解析地点
    auto parsed_location = request.explicit_location.has_value()
        ? *request.explicit_location
        : parseLocation(request.raw_text);
    
    // 5. 创建概念和记忆
    std::vector<ConceptId> concept_bindings;
    for (const auto& entity : entities) {
        ConceptType ctype = ConceptType::CONCEPT;
        if (entity.label == "PERSON") ctype = ConceptType::PERSON;
        else if (entity.label == "PLACE") ctype = ConceptType::PLACE;
        else if (entity.label == "EVENT") ctype = ConceptType::EVENT;
        else if (entity.label == "ORGANIZATION") ctype = ConceptType::ORGANIZATION;
        
        auto cid = getOrCreateConcept(entity.normalized_form, ctype, entity.confidence);
        concept_bindings.push_back(cid);
        
        // 检查是否是新概念
        auto existing = ontology_->getConcept(cid);
        if (existing && existing->evidence_count <= 1) {
            result.new_concepts_created++;
        } else {
            result.existing_concepts_linked++;
        }
    }
    
    // 6. 创建记忆
    MemoryTrace memory;
    memory.id = generateUUID();
    memory.timestamp = parsed_time;
    memory.location = parsed_location;
    memory.content.raw_text = request.raw_text;
    memory.content.embedding = embedding;
    memory.ontology_bindings = std::set<ConceptId>(
        concept_bindings.begin(), concept_bindings.end()
    );
    memory.source = request.source.value_or("user_input");
    
    // 7. 存储记忆
    memory_->store(memory);
    result.memory_id = memory.id;
    
    // 8. 绑定记忆到概念
    for (const auto& cid : concept_bindings) {
        ontology_->bindMemoryToConcept(cid, memory.id);
    }
    
    // 9. 执行推理
    result.reasoning_results = reasoning_->infer(std::nullopt, parsed_time);
    
    // 10. 生成推荐
    SituationContext ctx;
    ctx.current_time = parsed_time;
    if (parsed_location) ctx.current_location = *parsed_location;
    result.recommendations = generateRecommendationsInternal(ctx);
    
    return result;
}

// 智能查询
QueryResult POSCoreEngine::query(const QueryRequest& request) {
    QueryResult result;
    
    // 1. 提取查询实体
    auto entities = extractEntities(request.query_text);
    
    // 2. 确定查询策略
    std::string strategy = request.query_type;
    if (strategy == "auto") {
        bool has_time = false, has_location = false, has_person = false;
        for (const auto& e : entities) {
            if (e.label == "TIME" || e.label == "DATE") has_time = true;
            if (e.label == "PLACE") has_location = true;
            if (e.label == "PERSON") has_person = true;
        }
        
        if (has_person) strategy = "concept";
        else if (has_time) strategy = "temporal";
        else if (has_location) strategy = "spatial";
        else strategy = "semantic";
    }
    result.strategy_used = strategy;
    
    // 3. 执行查询
    if (strategy == "concept") {
        // 基于概念的查询
        for (const auto& entity : entities) {
            if (entity.label == "PERSON" || entity.label == "PLACE") {
                auto concepts = ontology_->findConceptsByLabel(entity.normalized_form, 1);
                for (const auto& c : concepts) {
                    auto memories = memory_->retrieveByConcepts({c.id}, false, 10);
                    for (const auto& mr : memories) {
                        result.direct_matches.push_back(mr.memory);
                    }
                    
                    // 推理扩展
                    auto inferred = reasoning_->inferTransitiveRelations(c.id);
                    for (const auto& inf : inferred) {
                        result.reasoning_path.push_back(inf);
                    }
                }
            }
        }
    }
    else if (strategy == "temporal" && request.time_hint) {
        // 时间查询
        auto start = *request.time_hint - std::chrono::hours(12);
        auto end = *request.time_hint + std::chrono::hours(12);
        auto memories = memory_->retrieveByTime(start, end, 20);
        for (const auto& mr : memories) {
            result.direct_matches.push_back(mr.memory);
        }
    }
    else if (strategy == "spatial" && request.location_hint) {
        // 空间查询
        auto memories = memory_->retrieveByLocation(*request.location_hint, 5000.0, 20);
        for (const auto& mr : memories) {
            result.direct_matches.push_back(mr.memory);
        }
    }
    else {
        // 语义查询
        auto dummy_embedding = generateEmbedding(request.query_text);
        auto memories = memory_->retrieveByText(request.query_text, dummy_embedding, 10);
        for (const auto& mr : memories) {
            result.direct_matches.push_back(mr.memory);
        }
    }
    
    return result;
}

// 生成推荐
std::vector<UnifiedProcessResult::Recommendation> 
POSCoreEngine::generateRecommendations(const SituationContext& context, int limit) {
    return generateRecommendationsInternal(context);
}

std::vector<UnifiedProcessResult::Recommendation> 
POSCoreEngine::generateRecommendationsInternal(const SituationContext& context) {
    std::vector<UnifiedProcessResult::Recommendation> recommendations;
    
    auto now = context.current_time;
    auto hour = std::chrono::system_clock::to_time_t(now) % 86400 / 3600 + 8; // 粗略转换为小时 (UTC+8)
    
    // 基于时间的推荐
    if (hour >= 6 && hour < 9) {
        // 早晨推荐
        recommendations.push_back({
            "habit",
            "早晨习惯",
            "根据您的历史记录，您经常在早晨处理重要事务",
            0.75f,
            3,
            "基于时间模式分析"
        });
    }
    
    // 基于位置的推荐
    if (context.current_location) {
        auto nearby = memory_->retrieveByLocation(*context.current_location, 1000.0, 5);
        if (!nearby.empty()) {
            recommendations.push_back({
                "location",
                "附近记忆",
                "您在这个位置附近有 " + std::to_string(nearby.size()) + " 条相关记忆",
                0.8f,
                3,
                "基于空间邻近性"
            });
        }
    }
    
    // 基于推理的推荐
    auto inferences = reasoning_->infer(std::nullopt, now);
    for (const auto& inf : inferences) {
        if (inf.confidence > 0.6) {
            recommendations.push_back({
                inf.type,
                inf.rule_name,
                inf.description,
                inf.confidence,
                4,
                "基于本体推理"
            });
        }
    }
    
    return recommendations;
}

// 预测未来事件
std::vector<InferenceResult> POSCoreEngine::predictNextEvents(
    const Timestamp& from_time, 
    int horizon_hours) {
    
    return reasoning_->predictNextEvents(from_time, horizon_hours);
}

// 实体提取
std::vector<ExtractedEntity> POSCoreEngine::extractEntities(const std::string& text) {
    std::vector<ExtractedEntity> result;
    
    try {
        auto entities = ml_client_->extractEntities(text);
        for (const auto& e : entities) {
            ExtractedEntity ee;
            ee.text = e.text;
            ee.label = e.label;
            ee.start_pos = e.start_pos;
            ee.end_pos = e.end_pos;
            ee.confidence = e.confidence;
            ee.normalized_form = disambiguateEntity(e.text, e.label);
            ee.canonical_id = "concept_" + ee.normalized_form;  // 简化处理
            result.push_back(ee);
        }
    } catch (const std::exception& e) {
        std::cerr << "[POSCore] Entity extraction failed: " << e.what() << std::endl;
    }
    
    return result;
}

// 生成embedding
Embedding POSCoreEngine::generateEmbedding(const std::string& text) {
    try {
        return ml_client_->embed(text);
    } catch (...) {
        // 返回零向量作为fallback
        return Embedding(384, 0.0f);
    }
}

// 获取或创建概念
ConceptId POSCoreEngine::getOrCreateConcept(const std::string& label,
                                            ConceptType type,
                                            float confidence) {
    auto existing = ontology_->findConceptsByLabel(label, 1);
    if (!existing.empty()) {
        return existing[0].id;
    }
    
    // 创建新概念
    auto source = confidence > 0.8 ? "auto_extracted_high_conf" : "auto_extracted";
    return ontology_->createConcept(label, type, source);
}

// 实体消歧
std::string POSCoreEngine::disambiguateEntity(const std::string& text,
                                               const std::string& label) {
    // 简单的别名映射
    static std::map<std::string, std::string> aliases = {
        {"中伟", "中伟"},
        {"张伟", "中伟"},
        {"李四", "李四"},
        {"老李", "李四"},
        {"星巴克", "星巴克"},
        {"星爸爸", "星巴克"}
    };
    
    auto it = aliases.find(text);
    if (it != aliases.end()) {
        return it->second;
    }
    return text;
}

// 获取统计
POSCoreEngine::Stats POSCoreEngine::getStats() const {
    Stats s;
    s.memory_count = memory_->getMemoryCount();
    s.concept_count = ontology_->getConceptCount();
    // s.relation_count = ...
    return s;
}

} // namespace pos
