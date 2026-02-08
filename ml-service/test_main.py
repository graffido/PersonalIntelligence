#!/usr/bin/env python3
"""简化版ML服务，用于测试验证"""

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import numpy as np

app = FastAPI(title="POS ML Service (Test)")

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

@app.get("/health")
def health():
    return {"status": "ok", "models": {"embedding": True, "ner": True}}

@app.post("/embed", response_model=EmbedResponse)
def embed(req: EmbedRequest):
    """生成随机embedding用于测试"""
    embeddings = [np.random.randn(384).tolist() for _ in req.texts]
    return EmbedResponse(
        embeddings=embeddings,
        model="test-embedding",
        dimension=384
    )

@app.post("/ner")
def ner(req: NERRequest):
    """简单的规则NER用于测试"""
    entities = []
    # 简单规则：检测大写字母开头的单词作为人名
    words = req.text.split()
    pos = 0
    for word in words:
        if word[0].isupper() and len(word) > 2:
            entities.append({
                "text": word,
                "label": "PERSON",
                "start": pos,
                "end": pos + len(word),
                "confidence": 0.9
            })
        pos += len(word) + 1
    return {"entities": entities, "model": "test-ner"}

@app.post("/generate")
def generate(req: dict):
    return {"text": f"[Test Response] {req.get('prompt', '')[:50]}...", "model": "test"}

if __name__ == "__main__":
    import uvicorn
    print("Starting test ML service on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
