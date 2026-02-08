#!/usr/bin/env python3
"""
POS 推理服务 API
提供基于本体的本地推理能力
"""

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Optional
from datetime import datetime
import os

from reasoning_engine import LocalReasoningEngine, InferenceResult, Concept, Memory

app = FastAPI(title="POS Reasoning Service", version="1.0.0")

# 全局推理引擎
engine = LocalReasoningEngine()

# ============ 请求/响应模型 ============

class ReasoningRequest(BaseModel):
    query: str
    entities: List[Dict]
    context: Optional[Dict] = None

class ReasoningResponse(BaseModel):
    direct_matches: List[Dict]
    inferred_matches: List[Dict]
    suggestions: List[Dict]
    reasoning_path: List[str]
    used_llm: bool = False  # 标记是否使用了LLM

class ConflictCheckRequest(BaseModel):
    time_window_hours: int = 48

class ConflictResponse(BaseModel):
    conflicts: List[Dict]
    count: int

class PatternDiscoveryRequest(BaseModel):
    min_frequency: int = 2

class PatternResponse(BaseModel):
    patterns: List[Dict]
    count: int

# ============ API端点 ============

@app.get("/health")
def health():
    """健康检查"""
    return {
        "status": "ok",
        "service": "reasoning",
        "capabilities": [
            "transitive_inference",
            "conflict_detection",
            "pattern_discovery",
            "local_query"
        ]
    }

@app.post("/reason", response_model=ReasoningResponse)
def reason(req: ReasoningRequest):
    """
    执行本地推理
    
    这是核心推理端点，优先使用本体推理，仅在必要时回退到LLM
    """
    # 使用本地推理引擎
    result = engine.query_with_reasoning(req.query, req.entities)
    
    # 添加推理路径说明
    result["reasoning_path"] = [
        "1. 实体识别: 从查询中提取关键实体",
        "2. 直接匹配: 在本体中查找精确匹配",
        "3. 关系扩展: 基于本体关系进行推理",
        "4. 模式匹配: 应用已发现的行为模式",
        "5. 冲突检测: 检查时间和空间约束"
    ]
    
    # 决定是否需要LLM增强
    needs_llm = len(result["direct_matches"]) == 0 and len(result["inferred_matches"]) == 0
    
    return ReasoningResponse(
        direct_matches=result["direct_matches"],
        inferred_matches=result["inferred_matches"],
        suggestions=result["suggestions"],
        reasoning_path=result["reasoning_path"],
        used_llm=needs_llm
    )

@app.post("/detect-conflicts", response_model=ConflictResponse)
def detect_conflicts(req: ConflictCheckRequest):
    """检测日程冲突"""
    conflicts = engine.detect_conflicts(req.time_window_hours)
    
    return ConflictResponse(
        conflicts=[{
            "type": c.result_type,
            "description": c.description,
            "confidence": c.confidence,
            "severity": c.metadata.get("severity", "medium"),
            "suggested_action": c.suggested_action
        } for c in conflicts],
        count=len(conflicts)
    )

@app.post("/discover-patterns", response_model=PatternResponse)
def discover_patterns(req: PatternDiscoveryRequest):
    """发现行为模式"""
    patterns = engine.discover_patterns(req.min_frequency)
    
    return PatternResponse(
        patterns=[{
            "type": p.pattern_type,
            "description": p.description,
            "confidence": p.confidence,
            "frequency": p.frequency,
            "prediction": p.prediction,
            "next_occurrence": p.next_occurrence.isoformat() if p.next_occurrence else None
        } for p in patterns],
        count=len(patterns)
    )

@app.post("/infer-concept/{concept_id}")
def infer_concept(concept_id: str):
    """针对特定概念进行推理"""
    results = engine.infer_for_concept(concept_id)
    
    return {
        "concept_id": concept_id,
        "inferences": [
            {
                "rule": r.rule_name,
                "type": r.result_type,
                "description": r.description,
                "confidence": r.confidence,
                "action": r.suggested_action
            }
            for r in results
        ],
        "count": len(results)
    }

@app.post("/add-concept")
def add_concept(concept: Dict):
    """添加概念到推理引擎"""
    c = Concept(
        id=concept["id"],
        type=concept["type"],
        label=concept["label"],
        properties=concept.get("properties", {}),
        relations=concept.get("relations", [])
    )
    engine.add_concept(c)
    return {"status": "ok", "concept_id": c.id}

@app.post("/add-memory")
def add_memory(memory: Dict):
    """添加记忆到推理引擎"""
    m = Memory(
        id=memory["id"],
        content=memory["content"],
        timestamp=datetime.fromisoformat(memory["timestamp"]),
        entities=memory.get("entities", []),
        location=memory.get("location"),
        emotions=memory.get("emotions", []),
        ontology_bindings=memory.get("ontology_bindings", [])
    )
    engine.add_memory(m)
    return {"status": "ok", "memory_id": m.id}

@app.get("/stats")
def stats():
    """获取推理引擎统计"""
    return {
        "concepts": len(engine.concepts),
        "memories": len(engine.memories),
        "rules": len(engine.rules)
    }

# 与主ML服务集成的辅助函数
def should_use_llm(query: str, local_results: ReasoningResponse) -> bool:
    """
    判断是否需要调用LLM
    
    当本地推理无法给出满意答案时，才调用LLM
    """
    # 如果有高质量的直接匹配，不需要LLM
    if any(m.get("confidence", 0) > 0.8 for m in local_results.direct_matches):
        return False
    
    # 如果有合理的推理匹配，不需要LLM
    if len(local_results.inferred_matches) >= 2:
        return False
    
    # 如果有有用的建议，不需要LLM
    if any(s.get("confidence", 0) > 0.7 for s in local_results.suggestions):
        return False
    
    # 查询过于复杂，需要LLM
    if len(query) > 100 or "?" in query or "为什么" in query or "怎么" in query:
        return True
    
    return True

if __name__ == "__main__":
    import uvicorn
    
    print("=" * 50)
    print("POS Reasoning Service")
    print("基于本体的本地推理引擎")
    print("=" * 50)
    print("\n端点:")
    print("  - GET  /health")
    print("  - POST /reason           - 执行推理")
    print("  - POST /detect-conflicts - 冲突检测")
    print("  - POST /discover-patterns - 模式发现")
    print("  - GET  /stats")
    print("=" * 50)
    
    uvicorn.run(app, host="0.0.0.0", port=8001)
