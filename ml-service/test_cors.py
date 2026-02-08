#!/usr/bin/env python3
"""
POS 增强版ML服务 - 支持中文实体识别 + CORS
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
import re

app = FastAPI(title="POS ML Service (Enhanced)")

# 启用CORS - 允许所有来源
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class EmbedRequest(BaseModel):
    texts: List[str]

class EmbedResponse(BaseModel):
    embeddings: List[List[float]]
    model: str
    dimension: int

class NERRequest(BaseModel):
    text: str

class Entity(BaseModel):
    text: str
    label: str
    start: int
    end: int
    confidence: float

# 中文实体识别规则
PERSON_NAMES = ["张三", "李四", "王五", "中伟", "李医生", "女朋友", "老板", "同事"]
PLACES = ["星巴克", "海底捞", "健身房", "医院", "公司", "咖啡厅", "餐厅", "电影院"]
ORGANIZATIONS = ["腾讯", "阿里", "字节", "百度", "公司"]

def extract_entities_rule(text: str) -> List[Entity]:
    """基于规则的实体抽取（支持中文）"""
    entities = []
    
    # 识别人名
    for name in PERSON_NAMES:
        for match in re.finditer(re.escape(name), text):
            entities.append(Entity(
                text=name,
                label="PERSON",
                start=match.start(),
                end=match.end(),
                confidence=0.95
            ))
    
    # 识别地点
    for place in PLACES:
        for match in re.finditer(re.escape(place), text):
            entities.append(Entity(
                text=place,
                label="GPE",
                start=match.start(),
                end=match.end(),
                confidence=0.90
            ))
    
    # 识别组织
    for org in ORGANIZATIONS:
        for match in re.finditer(re.escape(org), text):
            entities.append(Entity(
                text=org,
                label="ORG",
                start=match.start(),
                end=match.end(),
                confidence=0.85
            ))
    
    # 识别时间模式
    time_patterns = [
        (r'\d{4}年\d{1,2}月\d{1,2}日', 'DATE'),
        (r'\d{1,2}月\d{1,2}日', 'DATE'),
        (r'(?:早晨|上午|中午|下午|晚上|凌晨)\d{1,2}点', 'TIME'),
        (r'\d{1,2}点\d{1,2}分', 'TIME'),
        (r'昨天|今天|明天|后天', 'DATE'),
        (r'周一|周二|周三|周四|周五|周六|周日|星期[一二三四五六日]', 'DATE'),
    ]
    
    for pattern, label in time_patterns:
        for match in re.finditer(pattern, text):
            entities.append(Entity(
                text=match.group(),
                label=label,
                start=match.start(),
                end=match.end(),
                confidence=0.88
            ))
    
    return entities

@app.get("/health")
def health():
    return {"status": "ok", "models": {"embedding": True, "ner": True, "version": "enhanced-cors"}}

@app.post("/embed", response_model=EmbedResponse)
def embed(req: EmbedRequest):
    """生成embedding（模拟）"""
    embeddings = [np.random.randn(384).tolist() for _ in req.texts]
    return EmbedResponse(
        embeddings=embeddings,
        model="text-embedding-3-small",
        dimension=384
    )

@app.post("/ner")
def ner(req: NERRequest):
    """命名实体识别（中文支持）"""
    entities = extract_entities_rule(req.text)
    return {"entities": [e.dict() for e in entities], "model": "rule-based-zh", "text": req.text}

@app.post("/extract-relations")
def extract_relations(req: dict):
    """关系抽取"""
    return {"relations": []}

@app.post("/generate")
def generate(req: dict):
    """文本生成（模拟）"""
    prompt = req.get("prompt", "")
    return {
        "text": f"基于您的个人记忆，我找到了以下相关信息:\n\n关于'{prompt[:20]}...'的记忆:\n- 时间: 2024年1月\n- 地点: 星巴克\n- 参与人: 中伟\n\n这是您第3次在星巴克讨论项目。",
        "model": "gpt-4o-mini"
    }

@app.post("/process")
def process(req: dict):
    """完整处理流程"""
    text = req.get("text", "")
    entities = extract_entities_rule(text)
    embedding = np.random.randn(384).tolist()
    
    return {
        "entities": [e.dict() for e in entities],
        "embedding": embedding,
        "summary": text[:50] + "..." if len(text) > 50 else text,
        "model_info": {
            "embedding": "text-embedding-3-small",
            "ner": "rule-based-zh"
        }
    }

if __name__ == "__main__":
    import uvicorn
    print("=" * 50)
    print("POS ML Service (Enhanced + CORS)")
    print("支持中文实体识别")
    print("=" * 50)
    print("端点:")
    print("  - GET  /health")
    print("  - POST /embed")
    print("  - POST /ner")
    print("  - POST /process")
    print("=" * 50)
    uvicorn.run(app, host="0.0.0.0", port=8000)
