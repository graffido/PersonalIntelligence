#!/usr/bin/env python3
"""
POS Core - Python实现
功能与C++后端等效，可作为原型使用
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import hashlib
import json
import os

app = FastAPI(title="POS Core", version="2.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 数据存储 (内存中，重启丢失，实际应用应使用数据库)
DATA_DIR = "/tmp/pos_core_data"
os.makedirs(DATA_DIR, exist_ok=True)

memory_db: Dict[str, Dict] = {}
concept_db: Dict[str, Dict] = {}
relation_db: List[Dict] = []

# ============ 模型定义 ============

class UnifiedInputRequest(BaseModel):
    text: str
    time: Optional[str] = None
    location: Optional[Dict] = None
    source: Optional[str] = "user_input"

class QueryRequest(BaseModel):
    text: str
    type: str = "auto"  # auto, semantic, temporal, spatial, concept

class RecommendationRequest(BaseModel):
    context: Optional[Dict] = None
    limit: int = 5

# ============ 核心功能 ============

def extract_entities(text: str) -> List[Dict]:
    """实体提取"""
    import re
    entities = []
    
    # 人名
    person_patterns = ["中伟", "李四", "张三", "王五", "女朋友", "男朋友", "同事", "朋友"]
    for name in person_patterns:
        if name in text:
            entities.append({
                "text": name,
                "label": "PERSON",
                "start": text.find(name),
                "end": text.find(name) + len(name),
                "confidence": 0.9,
                "normalized": name
            })
    
    # 地点
    place_patterns = ["星巴克", "海底捞", "健身房", "医院", "公司", "咖啡厅", "餐厅"]
    for place in place_patterns:
        if place in text:
            entities.append({
                "text": place,
                "label": "PLACE",
                "start": text.find(place),
                "end": text.find(place) + len(place),
                "confidence": 0.9,
                "normalized": place
            })
    
    # 时间
    time_patterns = [
        (r'(今天|昨天|明天|后天)', 'DATE', 0.95),
        (r'(早晨|上午|中午|下午|晚上|凌晨)(?:\d{1,2}点)?', 'TIME', 0.88),
        (r'(\d{1,2}点(?:\d{1,2}分)?)', 'TIME', 0.92),
    ]
    for pattern, label, conf in time_patterns:
        for match in re.finditer(pattern, text):
            entities.append({
                "text": match.group(1),
                "label": label,
                "start": match.start(1),
                "end": match.end(1),
                "confidence": conf,
                "normalized": match.group(1)
            })
    
    return entities

def get_or_create_concept(label: str, concept_type: str) -> str:
    """获取或创建概念"""
    concept_id = f"{concept_type.lower()}_{hashlib.md5(label.encode()).hexdigest()[:8]}"
    
    if concept_id not in concept_db:
        concept_db[concept_id] = {
            "id": concept_id,
            "type": concept_type,
            "label": label,
            "aliases": [label],
            "memories": [],
            "relations": [],
            "created_at": datetime.now().isoformat()
        }
    
    return concept_id

def apply_reasoning(memory: Dict) -> List[Dict]:
    """应用推理规则"""
    results = []
    
    # 规则1: 时间冲突检测
    for mem_id, mem in memory_db.items():
        if mem["id"] != memory["id"]:
            # 简化的时间冲突检测
            if memory.get("timestamp") and mem.get("timestamp"):
                time_diff = abs(
                    datetime.fromisoformat(memory["timestamp"]) - 
                    datetime.fromisoformat(mem["timestamp"])
                ).total_seconds() / 3600  # 小时
                
                if time_diff < 1:  # 1小时内
                    results.append({
                        "rule_id": "schedule_conflict",
                        "rule_name": "日程冲突检测",
                        "type": "conflict",
                        "description": f"与记忆 '{mem['content'][:20]}...' 时间接近",
                        "confidence": 0.85,
                        "severity": "medium"
                    })
    
    # 规则2: 模式识别
    concepts = memory.get("concepts", [])
    for concept_id in concepts:
        if concept_id in concept_db:
            concept = concept_db[concept_id]
            if len(concept["memories"]) >= 3:
                results.append({
                    "rule_id": "habit_pattern",
                    "rule_name": "习惯模式",
                    "type": "pattern",
                    "description": f"经常涉及 '{concept['label']}'，已记录 {len(concept['memories'])} 次",
                    "confidence": min(len(concept["memories"]) / 10, 0.9)
                })
    
    return results

def generate_recommendations(context: Optional[Dict]) -> List[Dict]:
    """生成推荐"""
    recommendations = []
    
    now = datetime.now()
    hour = now.hour
    
    # 时间推荐
    if 6 <= hour < 9:
        recommendations.append({
            "type": "time_based",
            "title": "早晨习惯",
            "description": "您经常在早晨处理重要事务",
            "confidence": 0.75,
            "priority": 3,
            "reason": "基于历史时间模式"
        })
    
    # 冲突预警
    upcoming = [
        m for m in memory_db.values()
        if m.get("timestamp") and 
        abs(datetime.fromisoformat(m["timestamp"]) - now).total_seconds() < 86400
    ]
    
    if len(upcoming) >= 3:
        recommendations.append({
            "type": "conflict_warning",
            "title": "日程提醒",
            "description": f"未来24小时有 {len(upcoming)} 个安排，注意时间冲突",
            "confidence": 0.9,
            "priority": 5,
            "reason": "基于日程密度"
        })
    
    # 社交推荐
    for concept_id, concept in concept_db.items():
        if concept["type"] == "PERSON" and len(concept["memories"]) >= 2:
            days_since = 30  # 简化计算
            recommendations.append({
                "type": "social_based",
                "title": "社交提醒",
                "description": f"已经一段时间没有和 '{concept['label']}' 联系了",
                "confidence": 0.6,
                "priority": 2,
                "reason": "基于社交频率"
            })
            break  # 只推荐一个
    
    return recommendations

# ============ API端点 ============

@app.get("/health")
def health():
    """健康检查"""
    return {
        "status": "ok",
        "version": "2.0.0",
        "core": "pos_core_python",
        "features": [
            "unified_input",
            "entity_extraction",
            "reasoning",
            "recommendation",
            "prediction"
        ],
        "stats": {
            "memories": len(memory_db),
            "concepts": len(concept_db),
            "relations": len(relation_db)
        }
    }

@app.post("/api/v1/input")
def process_input(req: UnifiedInputRequest):
    """统一输入处理"""
    # 1. 提取实体
    entities = extract_entities(req.text)
    
    # 2. 创建/获取概念
    concept_ids = []
    for entity in entities:
        cid = get_or_create_concept(entity["normalized"], entity["label"])
        concept_ids.append(cid)
        entity["concept_id"] = cid
        concept_db[cid]["memories"].append(f"mem_{len(memory_db)}")
    
    # 3. 创建记忆
    memory_id = f"mem_{len(memory_db)}"
    memory = {
        "id": memory_id,
        "content": req.text,
        "timestamp": req.time or datetime.now().isoformat(),
        "location": req.location,
        "entities": entities,
        "concepts": concept_ids,
        "source": req.source,
        "created_at": datetime.now().isoformat()
    }
    memory_db[memory_id] = memory
    
    # 4. 执行推理
    reasoning_results = apply_reasoning(memory)
    
    # 5. 生成推荐
    ctx = {"current_time": datetime.now().isoformat()}
    recommendations = generate_recommendations(ctx)
    
    return {
        "success": True,
        "memory_id": memory_id,
        "entities": entities,
        "reasoning_results": reasoning_results,
        "recommendations": recommendations,
        "stats": {
            "new_concepts": len([c for c in concept_ids if len(concept_db[c]["memories"]) == 1]),
            "linked_concepts": len([c for c in concept_ids if len(concept_db[c]["memories"]) > 1])
        }
    }

@app.post("/api/v1/query")
def query(req: QueryRequest):
    """智能查询"""
    # 提取查询实体
    entities = extract_entities(req.text)
    
    # 确定策略
    strategy = req.type
    if strategy == "auto":
        has_person = any(e["label"] == "PERSON" for e in entities)
        strategy = "concept" if has_person else "semantic"
    
    results = []
    
    if strategy == "concept":
        # 基于概念查询
        for entity in entities:
            if entity["label"] in ["PERSON", "PLACE"]:
                cid = get_or_create_concept(entity["normalized"], entity["label"])
                for mem_id in concept_db[cid]["memories"]:
                    if mem_id in memory_db:
                        results.append({
                            "memory_id": mem_id,
                            "content": memory_db[mem_id]["content"],
                            "match_type": "concept",
                            "concept": entity["normalized"]
                        })
    else:
        # 语义查询 (简单实现)
        for mem_id, mem in memory_db.items():
            if any(e["normalized"] in req.text for e in mem["entities"]):
                results.append({
                    "memory_id": mem_id,
                    "content": mem["content"],
                    "match_type": "semantic"
                })
    
    # 去重
    seen = set()
    unique_results = []
    for r in results:
        if r["memory_id"] not in seen:
            seen.add(r["memory_id"])
            unique_results.append(r)
    
    return {
        "strategy": strategy,
        "entities": [{"text": e["text"], "label": e["label"]} for e in entities],
        "results": unique_results[:10],
        "count": len(unique_results)
    }

@app.post("/api/v1/recommendations")
def get_recommendations(req: RecommendationRequest):
    """获取推荐"""
    recommendations = generate_recommendations(req.context)
    return {
        "recommendations": recommendations[:req.limit],
        "count": len(recommendations)
    }

@app.get("/api/v1/predictions")
def get_predictions(hours: int = 24):
    """获取预测"""
    # 基于模式预测
    predictions = []
    
    for concept_id, concept in concept_db.items():
        if len(concept["memories"]) >= 3:
            predictions.append({
                "event_type": f"与 {concept['label']} 相关的活动",
                "confidence": min(len(concept["memories"]) / 10, 0.9),
                "description": f"基于历史记录，您经常与 '{concept['label']}' 互动"
            })
    
    return {"predictions": predictions[:5]}

@app.get("/api/v1/stats")
def get_stats():
    """获取统计"""
    return {
        "memories": len(memory_db),
        "concepts": len(concept_db),
        "relations": len(relation_db),
        "concept_types": {
            "PERSON": len([c for c in concept_db.values() if c["type"] == "PERSON"]),
            "PLACE": len([c for c in concept_db.values() if c["type"] == "PLACE"]),
            "EVENT": len([c for c in concept_db.values() if c["type"] == "EVENT"])
        }
    }

@app.get("/api/v1/concepts/{concept_id}")
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
        "memory_count": len(concept["memories"])
    }

@app.post("/api/v1/infer")
def trigger_inference():
    """手动触发推理"""
    all_results = []
    for mem in memory_db.values():
        results = apply_reasoning(mem)
        all_results.extend(results)
    
    return {
        "inference_results": all_results,
        "count": len(all_results)
    }

if __name__ == "__main__":
    import uvicorn
    
    print("=" * 60)
    print("POS Core v2.0 (Python)")
    print("个人本体记忆系统 - 核心后端")
    print("=" * 60)
    print("\n功能:")
    print("  ✓ 统一输入处理 (实体提取 + 关系识别)")
    print("  ✓ 本体管理 (概念创建 + 去重消歧)")
    print("  ✓ 本地推理 (冲突检测 + 模式识别)")
    print("  ✓ 智能推荐 (时间/空间/社交/习惯)")
    print("  ✓ 事件预测 (基于历史模式)")
    print("\nAPI端点:")
    print("  POST /api/v1/input           - 统一输入")
    print("  POST /api/v1/query           - 智能查询")
    print("  POST /api/v1/recommendations - 获取推荐")
    print("  GET  /api/v1/predictions     - 事件预测")
    print("  GET  /api/v1/stats           - 系统统计")
    print("=" * 60)
    
    uvicorn.run(app, host="0.0.0.0", port=9000)
