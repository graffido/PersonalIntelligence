# POS ML Service
# Python ML服务，提供embedding、NER、文本生成等功能

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import numpy as np
import os
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="POS ML Service",
    description="Personal Ontology System ML Backend",
    version="1.0.0"
)

# 全局模型变量
embedding_model = None
ner_pipeline = None
llm_client = None

# ============ 配置 ============
class ModelConfig:
    """模型配置"""
    # Embedding模型配置
    EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "local")  # openai, local
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    
    # NER模型配置
    NER_MODEL = os.getenv("NER_MODEL", "en_core_web_sm")  # spaCy模型
    
    # LLM配置
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "local")
    LLM_MODEL = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
    
    # 模型参数
    EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "384"))
    MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "32"))

# ============ 请求/响应模型 ============
class EmbedRequest(BaseModel):
    texts: List[str] = Field(..., min_items=1, max_items=100)
    model: Optional[str] = None

class EmbedResponse(BaseModel):
    embeddings: List[List[float]]
    model: str
    dimension: int

class NERRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000)
    entity_types: Optional[List[str]] = None

class Entity(BaseModel):
    text: str
    label: str
    start: int
    end: int
    confidence: float = 1.0

class NERResponse(BaseModel):
    entities: List[Entity]
    model: str

class Relation(BaseModel):
    subject: str
    predicate: str
    object: str
    confidence: float

class RelationRequest(BaseModel):
    text: str
    entities: List[Entity]

class RelationResponse(BaseModel):
    relations: List[Relation]

class GenerateRequest(BaseModel):
    prompt: str = Field(..., max_length=4000)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(500, ge=1, le=2000)
    model: Optional[str] = None

class GenerateResponse(BaseModel):
    text: str
    model: str
    tokens_used: Optional[int] = None

class ProcessRequest(BaseModel):
    text: str = Field(..., max_length=10000)
    extract_entities: bool = True
    generate_embedding: bool = True
    generate_summary: bool = False

class ProcessResponse(BaseModel):
    entities: List[Entity] = []
    embedding: Optional[List[float]] = None
    summary: Optional[str] = None
    model_info: Dict[str, str]

# ============ 模型加载 ============
def load_embedding_model():
    """加载embedding模型"""
    global embedding_model
    
    if embedding_model is not None:
        return embedding_model
    
    if ModelConfig.EMBEDDING_PROVIDER == "openai":
        try:
            import openai
            if ModelConfig.OPENAI_API_KEY:
                openai.api_key = ModelConfig.OPENAI_API_KEY
            embedding_model = {"provider": "openai", "client": openai}
            logger.info("Loaded OpenAI embedding model")
        except ImportError:
            logger.warning("OpenAI not available, falling back to local model")
            ModelConfig.EMBEDDING_PROVIDER = "local"
    
    if ModelConfig.EMBEDDING_PROVIDER == "local":
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(ModelConfig.EMEDDING_MODEL)
            embedding_model = {"provider": "local", "model": model}
            logger.info(f"Loaded local embedding model: {ModelConfig.EMBED_MODEL}")
        except ImportError:
            logger.error("sentence-transformers not installed")
            raise
    
    return embedding_model

def load_ner_model():
    """加载NER模型"""
    global ner_pipeline
    
    if ner_pipeline is not None:
        return ner_pipeline
    
    try:
        import spacy
        ner_pipeline = spacy.load(ModelConfig.NER_MODEL)
        logger.info(f"Loaded NER model: {ModelConfig.NER_MODEL}")
    except ImportError:
        logger.error("spaCy not installed or model not found")
        logger.info("Please run: python -m spacy download en_core_web_sm")
        # 创建一个简单的基于规则的NER作为fallback
        ner_pipeline = None
    
    return ner_pipeline

def load_llm():
    """加载LLM"""
    global llm_client
    
    if llm_client is not None:
        return llm_client
    
    if ModelConfig.LLM_PROVIDER == "openai":
        try:
            import openai
            if ModelConfig.OPENAI_API_KEY:
                openai.api_key = ModelConfig.OPENAI_API_KEY
            llm_client = {"provider": "openai", "client": openai}
            logger.info("Loaded OpenAI LLM")
        except ImportError:
            logger.warning("OpenAI not available")
    
    return llm_client

# ============ API端点 ============
@app.on_event("startup")
async def startup_event():
    """启动时加载模型"""
    logger.info("Starting ML Service...")
    try:
        load_embedding_model()
    except Exception as e:
        logger.error(f"Failed to load embedding model: {e}")
    
    try:
        load_ner_model()
    except Exception as e:
        logger.error(f"Failed to load NER model: {e}")
    
    try:
        load_llm()
    except Exception as e:
        logger.error(f"Failed to load LLM: {e}")
    
    logger.info("ML Service ready")

@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "ok",
        "models": {
            "embedding": embedding_model is not None,
            "ner": ner_pipeline is not None,
            "llm": llm_client is not None
        },
        "config": {
            "embedding_provider": ModelConfig.EMBEDDING_PROVIDER,
            "embedding_model": ModelConfig.EMBEDDING_MODEL,
            "ner_model": ModelConfig.NER_MODEL
        }
    }

@app.post("/embed", response_model=EmbedResponse)
async def embed_texts(request: EmbedRequest):
    """生成文本embedding"""
    try:
        model = load_embedding_model()
        
        if model["provider"] == "openai":
            import openai
            response = openai.Embedding.create(
                input=request.texts,
                model="text-embedding-3-small"
            )
            embeddings = [item["embedding"] for item in response["data"]]
            return EmbedResponse(
                embeddings=embeddings,
                model="text-embedding-3-small",
                dimension=len(embeddings[0])
            )
        
        else:  # local
            embeddings = model["model"].encode(request.texts)
            embeddings_list = embeddings.tolist() if hasattr(embeddings, 'tolist') else embeddings
            return EmbedResponse(
                embeddings=embeddings_list,
                model=ModelConfig.EMBEDDING_MODEL,
                dimension=len(embeddings_list[0])
            )
    
    except Exception as e:
        logger.error(f"Embedding error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ner", response_model=NERResponse)
async def extract_entities(request: NERRequest):
    """命名实体识别"""
    try:
        nlp = load_ner_model()
        
        if nlp is None:
            # Fallback: 简单的规则匹配
            return NERResponse(entities=[], model="fallback")
        
        doc = nlp(request.text)
        entities = []
        
        for ent in doc.ents:
            # 过滤实体类型
            if request.entity_types and ent.label_ not in request.entity_types:
                continue
            
            entities.append(Entity(
                text=ent.text,
                label=ent.label_,
                start=ent.start_char,
                end=ent.end_char,
                confidence=1.0
            ))
        
        return NERResponse(entities=entities, model=ModelConfig.NER_MODEL)
    
    except Exception as e:
        logger.error(f"NER error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/extract-relations", response_model=RelationResponse)
async def extract_relations(request: RelationRequest):
    """关系抽取（简化实现）"""
    # 这里可以使用专门的关系抽取模型
    # 简化版本：基于规则/模板
    relations = []
    return RelationResponse(relations=relations)

@app.post("/generate", response_model=GenerateResponse)
async def generate_text(request: GenerateRequest):
    """文本生成"""
    try:
        llm = load_llm()
        
        if llm is None or llm.get("provider") != "openai":
            # Fallback: 返回简单响应
            return GenerateResponse(
                text="[LLM not available] " + request.prompt[:100],
                model="fallback",
                tokens_used=0
            )
        
        import openai
        response = openai.ChatCompletion.create(
            model=request.model or ModelConfig.LLM_MODEL,
            messages=[{"role": "user", "content": request.prompt}],
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )
        
        return GenerateResponse(
            text=response.choices[0].message.content,
            model=response.model,
            tokens_used=response.usage.total_tokens if response.usage else None
        )
    
    except Exception as e:
        logger.error(f"Generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process", response_model=ProcessResponse)
async def process_document(request: ProcessRequest):
    """完整文档处理"""
    result = ProcessResponse(model_info={})
    
    try:
        # 实体抽取
        if request.extract_entities:
            ner_result = await extract_entities(NERRequest(text=request.text))
            result.entities = ner_result.entities
            result.model_info["ner"] = ner_result.model
        
        # Embedding生成
        if request.generate_embedding:
            embed_result = await embed_texts(EmbedRequest(texts=[request.text]))
            result.embedding = embed_result.embeddings[0] if embed_result.embeddings else None
            result.model_info["embedding"] = embed_result.model
        
        # 摘要生成
        if request.generate_summary:
            summary_prompt = f"Summarize this in one sentence: {request.text[:500]}"
            gen_result = await generate_text(GenerateRequest(prompt=summary_prompt))
            result.summary = gen_result.text
            result.model_info["llm"] = gen_result.model
        
        return result
    
    except Exception as e:
        logger.error(f"Processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models")
async def list_models():
    """列出可用模型"""
    return {
        "embedding": {
            "current": ModelConfig.EMBEDDING_MODEL,
            "provider": ModelConfig.EMBEDDING_PROVIDER,
            "dimension": ModelConfig.EMBEDDING_DIM
        },
        "ner": {
            "current": ModelConfig.NER_MODEL,
            "available": ["en_core_web_sm", "en_core_web_md", "en_core_web_lg"]
        },
        "llm": {
            "current": ModelConfig.LLM_MODEL,
            "provider": ModelConfig.LLM_PROVIDER
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
