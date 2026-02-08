#!/usr/bin/env python3
"""
POS 统一API服务
整合: 输入解析、推理、预测推荐
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
from datetime import datetime
import uuid

from unified_input_parser import UnifiedInputParser, ParsedInput
from reasoning_engine import LocalReasoningEngine, Concept, Memory
from prediction_recommendation_engine import PredictionRecommendationEngine, Recommendation

app = FastAPI(title="POS Unified API", version="2.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化组件
input_parser = UnifiedInputParser()
reasoning_engine = LocalReasoningEngine()
prediction_engine = None  # 延迟初始化

# 内存存储 (实际应用应使用持久化存储)
memory_db: Dict[str, Dict] = {}
concept_db: Dict[str, Dict] = {}

# ============ 模型定义 ============

class UnifiedInputRequest(BaseModel):
    text: str
    context: Optional[Dict] = None

class UnifiedInputResponse(BaseModel):
    success: bool
    memory_id: str
    parsed: Dict
    entities: List[Dict]
    relations: List[Dict]
    reasoning_results: List[Dict]
    recommendations: List[Dict]

class QueryRequest(BaseModel):
    text: str
    query_type: str = "auto"  # auto, semantic, temporal, spatial, concept

class RecommendationRequest(BaseModel):
    context: Optional[Dict] = None
    limit: int = 5

# ============ API端点 ============

@app.get("/health")
def health():
    return {
        "status": "ok",
        "version": "2.0.0",
        "features": [
            "unified_input",
            "entity_extraction",
            "disambiguation",
            "reasoning",
            "prediction",
            "recommendation"
        ],
        "stats": {
            "memories": len(memory_db),
            "concepts": len(concept_db)
        }
    }

@app.post("/input", response_model=UnifiedInputResponse)
def process_unified_input(req: UnifiedInputRequest):
    """
    统一输入处理 - 单一入口处理所有输入
    """
    global prediction_engine
    if prediction_engine is None:
        prediction_engine = PredictionRecommendationEngine(
            memory_db, concept_db, reasoning_engine
        )
    
    # 1. 解析输入
    parsed = input_parser.parse(req.text, req.context)
    
    # 2. 创建记忆
    memory_id = f"mem_{uuid.uuid4().hex[:8]}"
    memory = Memory(
        id=memory_id,
        content=parsed.raw_text,
        timestamp=parsed.timestamp or datetime.now(),
        entities=[{"text": e.text, "label": e.label, "confidence": e.confidence} for e in parsed.entities],
        location=parsed.location,
        ontology_bindings=[e.canonical_id for e in parsed.entities if e.canonical_id]
    )
    memory_db[memory_id] = memory
    
    # 3. 更新本体 (去重)
    for entity in parsed.entities:
        concept_id = entity.canonical_id
        if concept_id and concept_id not in concept_db:
            concept_db[concept_id] = {
                "id": concept_id,
                "type": entity.label,
                "label": entity.normalized_form or entity.text,
                "aliases": [entity.text],
                "memories": [memory_id]
            }
        elif concept_id:
            # 添加别名和关联记忆
            if entity.text not in concept_db[concept_id]["aliases"]:
                concept_db[concept_id]["aliases"].append(entity.text)
            if memory_id not in concept_db[concept_id]["memories"]:
                concept_db[concept_id]["memories"].append(memory_id)
    
    # 4. 添加关系到推理引擎
    for rel in parsed.relations:
        reasoning_engine.add_concept(Concept(
            id=rel.subject,
            type="PERSON",
            label=rel.subject,
            relations=[{"type": rel.predicate, "target": rel.object, "weight": rel.confidence}]
        ))
    
    # 5. 执行推理
    reasoning_results = reasoning_engine.infer()
    
    # 6. 生成推荐
    context = req.context or {}
    context['current_time'] = datetime.now()
    recommendations = prediction_engine.generate_recommendations(context)
    
    return UnifiedInputResponse(
        success=True,
        memory_id=memory_id,
        parsed={
            "text": parsed.raw_text,
            "timestamp": parsed.timestamp.isoformat() if parsed.timestamp else None,
            "location": parsed.location,
            "main_event": parsed.main_event,
            "sentiment": parsed.sentiment
        },
        entities=[{
            "text": e.text,
            "label": e.label,
            "normalized": e.normalized_form,
            "canonical_id": e.canonical_id,
            "confidence": e.confidence
        } for e in parsed.entities],
        relations=[{
            "subject": r.subject,
            "predicate": r.predicate,
            "object": r.object,
            "confidence": r.confidence
        } for r in parsed.relations],
        reasoning_results=[{
            "rule": r.rule_name,
            "type": r.result_type,
            "description": r.description,
            "confidence": r.confidence
        } for r in reasoning_results[:5]],
        recommendations=[{
            "type": r.type.value,
            "title": r.title,
            "description": r.description,
            "confidence": r.confidence,
            "priority": r.priority,
            "reason": r.reason
        } for r in recommendations]
    )

@app.post("/query")
def query(req: QueryRequest):
    """
    智能查询 - 自动选择查询策略
    """
    # 解析查询
    parsed = input_parser.parse(req.text)
    
    # 根据查询内容自动选择策略
    if req.query_type == "auto":
        if any(e.label in ["TIME", "DATE"] for e in parsed.entities):
            strategy = "temporal"
        elif any(e.label == "PLACE" for e in parsed.entities):
            strategy = "spatial"
        elif any(e.label == "PERSON" for e in parsed.entities):
            strategy = "concept"
        else:
            strategy = "semantic"
    else:
        strategy = req.query_type
    
    # 执行查询
    results = []
    
    if strategy == "concept":
        # 基于概念的查询
        for entity in parsed.entities:
            if entity.canonical_id and entity.canonical_id in concept_db:
                concept = concept_db[entity.canonical_id]
                for mem_id in concept["memories"]:
                    if mem_id in memory_db:
                        results.append({
                            "memory_id": mem_id,
                            "content": memory_db[mem_id].content,
                            "match_type": "concept",
                            "concept": concept["label"]
                        })
    
    elif strategy == "temporal":
        # 基于时间的查询
        if parsed.timestamp:
            # 查找同一天的记忆
            for mem_id, mem in memory_db.items():
                if mem.timestamp.date() == parsed.timestamp.date():
                    results.append({
                        "memory_id": mem_id,
                        "content": mem.content,
                        "match_type": "temporal",
                        "time": mem.timestamp.isoformat()
                    })
    
    # 去重
    seen = set()
    unique_results = []
    for r in results:
        if r["memory_id"] not in seen:
            seen.add(r["memory_id"])
            unique_results.append(r)
    
    return {
        "query": req.text,
        "strategy": strategy,
        "entities": [{"text": e.text, "label": e.label} for e in parsed.entities],
        "results": unique_results[:10],
        "count": len(unique_results)
    }

@app.post("/recommendations")
def get_recommendations(req: RecommendationRequest):
    """
    获取个性化推荐
    """
    global prediction_engine
    if prediction_engine is None:
        prediction_engine = PredictionRecommendationEngine(
            memory_db, concept_db, reasoning_engine
        )
    
    context = req.context or {}
    context['current_time'] = datetime.now()
    
    recommendations = prediction_engine.generate_recommendations(context, req.limit)
    
    return {
        "recommendations": [
            {
                "id": r.id,
                "type": r.type.value,
                "title": r.title,
                "description": r.description,
                "confidence": r.confidence,
                "priority": r.priority,
                "reason": r.reason,
                "suggested_action": r.suggested_action
            }
            for r in recommendations
        ],
        "context": context
    }

@app.get("/predictions")
def get_predictions(hours: int = 24):
    """
    获取未来事件预测
    """
    global prediction_engine
    if prediction_engine is None:
        prediction_engine = PredictionRecommendationEngine(
            memory_db, concept_db, reasoning_engine
        )
    
    predictions = prediction_engine.predict_next_events(
        datetime.now(),
        horizon_hours=hours
    )
    
    return {
        "predictions": [
            {
                "event_type": p.event_type,
                "predicted_time": p.predicted_time.isoformat(),
                "confidence": p.confidence,
                "explanation": p.explanation
            }
            for p in predictions
        ]
    }

@app.get("/stats")
def get_stats():
    """获取系统统计"""
    # 计算关系强度分布
    relationship_strengths = reasoning_engine._calculate_relationship_strengths()
    
    return {
        "memories": len(memory_db),
        "concepts": len(concept_db),
        "relationships": len(relationship_strengths),
        "concept_types": {
            "PERSON": len([c for c in concept_db.values() if c["type"] == "PERSON"]),
            "PLACE": len([c for c in concept_db.values() if c["type"] == "PLACE"]),
            "EVENT": len([c for c in concept_db.values() if c["type"] == "EVENT"])
        }
    }

@app.get("/concepts/{concept_id}")
def get_concept(concept_id: str):
    """获取概念详情"""
    if concept_id not in concept_db:
        return {"error": "Concept not found"}
    
    concept = concept_db[concept_id]
    return {
        "id": concept["id"],
        "type": concept["type"],
        "label": concept["label"],
        "aliases": concept["aliases"],
        "memories": len(concept["memories"]),
        "related_concepts": []  # 可通过关系推导
    }

if __name__ == "__main__":
    import uvicorn
    
    print("=" * 60)
    print("POS Unified API v2.0")
    print("统一输入处理 + 推理 + 预测推荐")
    print("=" * 60)
    print("\n端点:")
    print("  POST /input          - 统一输入处理")
    print("  POST /query          - 智能查询")
    print("  POST /recommendations - 获取推荐")
    print("  GET  /predictions    - 事件预测")
    print("  GET  /stats          - 系统统计")
    print("=" * 60)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
