"""
增强版ML服务主入口
整合NER、Embedding和LLM服务
"""
import os
import yaml
import json
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
import uvicorn

from enhanced_ner import (
    EnhancedNER, Entity, EntityType, RelationType, 
    NERResult, create_ner_service, quick_extract
)
from embedding_service import (
    EmbeddingService, EmbeddingResult, 
    create_embedding_service, quick_encode
)
from llm_service import (
    LLMService, Message, ModelType, TaskComplexity,
    create_llm_service, quick_chat
)


# 加载配置
def load_config(config_path: str = "config.yaml") -> Dict:
    """加载配置文件"""
    config_file = Path(config_path)
    
    if config_file.exists():
        with open(config_file, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    # 默认配置
    return {
        "server": {
            "host": "0.0.0.0",
            "port": 8000,
            "reload": False
        },
        "ner": {
            "use_bert_chinese": False,
            "use_transformer": True,
            "use_ontology": True,
            "ontology_path": "ontology_db"
        },
        "embedding": {
            "default_model": "all-MiniLM-L6-v2",
            "device": "auto",
            "use_cache": True,
            "normalize": True
        },
        "llm": {
            "default_model": "gpt-4o-mini",
            "default_provider": "openai",
            "use_cache": True,
            "cache_size": 1000,
            "routing": {
                "simple": "ollama",
                "moderate": "ollama",
                "complex": "openai"
            }
        }
    }


# 全局服务实例
ner_service: Optional[EnhancedNER] = None
embedding_service: Optional[EmbeddingService] = None
llm_service: Optional[LLMService] = None
service_config: Dict = {}


# Pydantic模型定义
class NERRequest(BaseModel):
    text: str = Field(..., description="输入文本")
    extract_relations: bool = Field(True, description="是否抽取关系")
    use_ontology: Optional[bool] = Field(None, description="是否使用Ontology增强")


class NERBatchRequest(BaseModel):
    texts: List[str] = Field(..., description="文本列表")
    extract_relations: bool = Field(False, description="是否抽取关系")


class EntityAddRequest(BaseModel):
    text: str = Field(..., description="实体文本")
    entity_type: str = Field(..., description="实体类型")
    metadata: Optional[Dict] = Field(None, description="元数据")


class SynonymAddRequest(BaseModel):
    synonym: str = Field(..., description="同义词")
    canonical: str = Field(..., description="标准词")


class EmbeddingRequest(BaseModel):
    texts: Union[str, List[str]] = Field(..., description="输入文本或文本列表")
    model: Optional[str] = Field(None, description="模型ID")
    batch_size: int = Field(32, description="批处理大小")


class SimilarityRequest(BaseModel):
    text1: str = Field(..., description="文本1")
    text2: str = Field(..., description="文本2")
    model: Optional[str] = Field(None, description="模型ID")


class SearchRequest(BaseModel):
    query: str = Field(..., description="查询文本")
    documents: List[Dict[str, Any]] = Field(..., description="文档列表")
    top_k: int = Field(5, description="返回数量")
    model: Optional[str] = Field(None, description="模型ID")


class ClusteringRequest(BaseModel):
    texts: List[str] = Field(..., description="文本列表")
    n_clusters: int = Field(5, description="聚类数量")
    model: Optional[str] = Field(None, description="模型ID")


class ChatRequest(BaseModel):
    messages: Union[str, List[Dict[str, str]]] = Field(..., description="消息或消息列表")
    provider: Optional[str] = Field(None, description="Provider类型")
    model: Optional[str] = Field(None, description="模型名称")
    temperature: float = Field(0.7, description="温度参数")
    max_tokens: int = Field(1024, description="最大token数")
    stream: bool = Field(False, description="是否流式输出")
    use_cache: bool = Field(True, description="是否使用缓存")


class ChatMessage(BaseModel):
    role: str = Field(..., description="角色: system/user/assistant")
    content: str = Field(..., description="消息内容")


# 生命周期管理
@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global ner_service, embedding_service, llm_service, service_config
    
    # 启动时加载配置和服务
    service_config = load_config()
    print(f"🚀 正在启动ML服务...")
    print(f"配置: {json.dumps(service_config, indent=2, default=str)}")
    
    # 初始化服务
    print("\n📦 正在初始化服务...")
    
    # NER服务
    try:
        ner_service = create_ner_service(service_config.get("ner", {}))
        ner_service.load_models()
        print("✅ NER服务已初始化")
    except Exception as e:
        print(f"⚠️ NER服务初始化失败: {e}")
    
    # Embedding服务
    try:
        embedding_service = create_embedding_service(service_config.get("embedding", {}))
        print("✅ Embedding服务已初始化")
    except Exception as e:
        print(f"⚠️ Embedding服务初始化失败: {e}")
    
    # LLM服务
    try:
        llm_service = create_llm_service(service_config.get("llm", {}))
        providers = llm_service.get_available_providers()
        print(f"✅ LLM服务已初始化，可用providers: {[p['type'] for p in providers]}")
    except Exception as e:
        print(f"⚠️ LLM服务初始化失败: {e}")
    
    print("\n🎉 所有服务初始化完成！")
    
    yield
    
    # 关闭时清理
    print("\n🛑 正在关闭服务...")
    if ner_service:
        ner_service.save_ontology()
        print("💾 Ontology数据已保存")
    print("👋 服务已关闭")


# 创建FastAPI应用
app = FastAPI(
    title="增强版ML服务",
    description="集成NER、Embedding和LLM的机器学习服务",
    version="2.0.0",
    lifespan=lifespan
)

# CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ========== NER API ==========

@app.post("/ner/extract", response_model=Dict)
async def ner_extract(request: NERRequest):
    """抽取命名实体"""
    if not ner_service:
        raise HTTPException(status_code=503, detail="NER服务不可用")
    
    try:
        result = ner_service.extract(
            text=request.text,
            extract_relations=request.extract_relations,
            use_ontology=request.use_ontology
        )
        return result.to_dict()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ner/extract_batch", response_model=List[Dict])
async def ner_extract_batch(request: NERBatchRequest):
    """批量抽取命名实体"""
    if not ner_service:
        raise HTTPException(status_code=503, detail="NER服务不可用")
    
    try:
        results = ner_service.extract_batch(
            texts=request.texts,
            extract_relations=request.extract_relations
        )
        return [r.to_dict() for r in results]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ner/ontology/entity")
async def ner_add_entity(request: EntityAddRequest):
    """添加实体到Ontology"""
    if not ner_service:
        raise HTTPException(status_code=503, detail="NER服务不可用")
    
    try:
        ner_service.add_to_ontology(
            text=request.text,
            entity_type=request.entity_type,
            metadata=request.metadata
        )
        return {"status": "success", "message": f"实体 '{request.text}' 已添加到Ontology"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ner/ontology/synonym")
async def ner_add_synonym(request: SynonymAddRequest):
    """添加同义词"""
    if not ner_service:
        raise HTTPException(status_code=503, detail="NER服务不可用")
    
    try:
        ner_service.add_synonym(request.synonym, request.canonical)
        return {"status": "success", "message": f"同义词映射已添加"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ner/ontology/stats")
async def ner_ontology_stats():
    """获取Ontology统计信息"""
    if not ner_service:
        raise HTTPException(status_code=503, detail="NER服务不可用")
    
    return ner_service.get_ontology_stats()


@app.get("/ner/entity/{text}")
async def ner_get_entity(text: str):
    """获取实体详细信息"""
    if not ner_service:
        raise HTTPException(status_code=503, detail="NER服务不可用")
    
    info = ner_service.get_entity_info(text)
    if info:
        return info
    raise HTTPException(status_code=404, detail=f"未找到实体: {text}")


@app.get("/ner/entity_types")
async def ner_entity_types():
    """获取支持的实体类型"""
    return {
        "types": [
            {"name": t.value, "description": get_entity_type_desc(t)}
            for t in EntityType
        ]
    }


def get_entity_type_desc(entity_type: EntityType) -> str:
    """获取实体类型描述"""
    descriptions = {
        EntityType.PERSON: "人名、人物",
        EntityType.PLACE: "地点、位置",
        EntityType.EVENT: "事件、活动",
        EntityType.CONCEPT: "概念、主题",
        EntityType.ORGANIZATION: "组织、公司、机构",
        EntityType.TIME: "时间、日期",
        EntityType.MONEY: "金额、货币",
        EntityType.PRODUCT: "产品、物品",
        EntityType.WORK_OF_ART: "艺术作品、书籍、电影",
        EntityType.CUSTOM: "自定义类型"
    }
    return descriptions.get(entity_type, "")


# ========== Embedding API ==========

@app.post("/embedding/encode")
async def embedding_encode(request: EmbeddingRequest):
    """编码文本为embedding向量"""
    if not embedding_service:
        raise HTTPException(status_code=503, detail="Embedding服务不可用")
    
    try:
        results = embedding_service.encode(
            texts=request.texts,
            model_id=request.model,
            batch_size=request.batch_size
        )
        
        if isinstance(results, list):
            return {
                "embeddings": [
                    {
                        "text": r.text[:100] + "..." if len(r.text) > 100 else r.text,
                        "embedding": r.embedding.tolist(),
                        "dimension": r.dimension,
                        "model": r.model
                    }
                    for r in results
                ],
                "count": len(results)
            }
        else:
            return {
                "embedding": results.embedding.tolist(),
                "dimension": results.dimension,
                "model": results.model
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/embedding/similarity")
async def embedding_similarity(request: SimilarityRequest):
    """计算文本相似度"""
    if not embedding_service:
        raise HTTPException(status_code=503, detail="Embedding服务不可用")
    
    try:
        similarity = embedding_service.similarity(
            text1=request.text1,
            text2=request.text2,
            model_id=request.model
        )
        return {
            "text1": request.text1,
            "text2": request.text2,
            "similarity": similarity,
            "model": request.model or embedding_service.default_model
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/embedding/search")
async def embedding_search(request: SearchRequest):
    """语义搜索"""
    if not embedding_service:
        raise HTTPException(status_code=503, detail="Embedding服务不可用")
    
    try:
        results = embedding_service.semantic_search(
            query=request.query,
            documents=request.documents,
            top_k=request.top_k,
            model_id=request.model
        )
        return {
            "query": request.query,
            "results": results,
            "total": len(results)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/embedding/cluster")
async def embedding_cluster(request: ClusteringRequest):
    """文本聚类"""
    if not embedding_service:
        raise HTTPException(status_code=503, detail="Embedding服务不可用")
    
    try:
        clusters = embedding_service.clustering(
            texts=request.texts,
            n_clusters=request.n_clusters,
            model_id=request.model
        )
        return {"clusters": clusters}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/embedding/models")
async def embedding_models():
    """获取可用模型列表"""
    if not embedding_service:
        raise HTTPException(status_code=503, detail="Embedding服务不可用")
    
    return {"models": embedding_service.list_models()}


@app.get("/embedding/cache/stats")
async def embedding_cache_stats():
    """获取缓存统计"""
    if not embedding_service:
        raise HTTPException(status_code=503, detail="Embedding服务不可用")
    
    return embedding_service.get_cache_stats() or {"enabled": False}


@app.delete("/embedding/cache")
async def embedding_clear_cache():
    """清空缓存"""
    if not embedding_service:
        raise HTTPException(status_code=503, detail="Embedding服务不可用")
    
    embedding_service.clear_cache()
    return {"status": "success", "message": "缓存已清空"}


# ========== LLM API ==========

@app.post("/llm/chat")
async def llm_chat(request: ChatRequest):
    """LLM聊天"""
    if not llm_service:
        raise HTTPException(status_code=503, detail="LLM服务不可用")
    
    try:
        # 处理消息格式
        if isinstance(request.messages, str):
            messages = [Message(role="user", content=request.messages)]
        else:
            messages = [Message(role=m["role"], content=m["content"]) for m in request.messages]
        
        # 选择provider
        provider = None
        if request.provider:
            provider = ModelType(request.provider)
        
        # 流式响应
        if request.stream:
            def generate():
                for chunk in llm_service.stream_chat(
                    messages=messages,
                    provider=provider,
                    model=request.model,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens
                ):
                    yield f"data: {json.dumps({'content': chunk})}\n\n"
                yield "data: [DONE]\n\n"
            
            return StreamingResponse(
                generate(),
                media_type="text/event-stream"
            )
        
        # 普通响应
        response = llm_service.chat(
            messages=messages,
            provider=provider,
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            use_cache=request.use_cache
        )
        
        return response.to_dict()
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/llm/chat/simple")
async def llm_chat_simple(message: str, provider: Optional[str] = None):
    """简化版聊天接口"""
    if not llm_service:
        raise HTTPException(status_code=503, detail="LLM服务不可用")
    
    try:
        provider_type = ModelType(provider) if provider else None
        content = llm_service.simple_chat(message, provider=provider_type)
        return {"content": content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/llm/providers")
async def llm_providers():
    """获取可用providers"""
    if not llm_service:
        raise HTTPException(status_code=503, detail="LLM服务不可用")
    
    return {"providers": llm_service.get_available_providers()}


@app.get("/llm/stats")
async def llm_stats():
    """获取LLM使用统计"""
    if not llm_service:
        raise HTTPException(status_code=503, detail="LLM服务不可用")
    
    return llm_service.get_stats()


@app.delete("/llm/cache")
async def llm_clear_cache():
    """清空LLM缓存"""
    if not llm_service:
        raise HTTPException(status_code=503, detail="LLM服务不可用")
    
    llm_service.clear_cache()
    return {"status": "success", "message": "缓存已清空"}


# ========== 系统API ==========

@app.get("/")
async def root():
    """根路径"""
    return {
        "service": "增强版ML服务",
        "version": "2.0.0",
        "docs": "/docs",
        "endpoints": {
            "ner": "/ner/*",
            "embedding": "/embedding/*",
            "llm": "/llm/*"
        }
    }


@app.get("/health")
async def health():
    """健康检查"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "ner": ner_service is not None,
            "embedding": embedding_service is not None,
            "llm": llm_service is not None and len(llm_service.get_available_providers()) > 0
        }
    }


@app.get("/config")
async def get_config():
    """获取当前配置（脱敏）"""
    safe_config = json.loads(json.dumps(service_config, default=str))
    
    # 移除敏感信息
    for section in ["openai", "anthropic"]:
        if section in safe_config.get("llm", {}):
            safe_config["llm"][section]["api_key"] = "***" if safe_config["llm"][section].get("api_key") else None
    
    return safe_config


# ========== 工具函数 ==========

def save_config(config: Dict, path: str = "config.yaml"):
    """保存配置到文件"""
    with open(path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False)


def create_default_config():
    """创建默认配置文件"""
    default_config = {
        "server": {
            "host": "0.0.0.0",
            "port": 8000,
            "reload": False
        },
        "ner": {
            "use_bert_chinese": False,
            "chinese_bert_model": "bert-base-chinese",
            "use_transformer": True,
            "english_model": "en_core_web_trf",
            "use_ontology": True,
            "ontology_path": "ontology_db"
        },
        "embedding": {
            "default_model": "all-MiniLM-L6-v2",
            "device": "auto",
            "use_cache": True,
            "normalize": True,
            "cache_dir": "embedding_cache"
        },
        "llm": {
            "default_model": "gpt-4o-mini",
            "default_provider": "openai",
            "use_cache": True,
            "cache_size": 1000,
            "cache_ttl": 3600,
            "openai": {
                "enabled": True,
                "model": "gpt-4o-mini",
                "api_key": os.getenv("OPENAI_API_KEY", ""),
                "base_url": "https://api.openai.com/v1",
                "temperature": 0.7,
                "max_tokens": 1024
            },
            "anthropic": {
                "enabled": True,
                "model": "claude-3-haiku-20240307",
                "api_key": os.getenv("ANTHROPIC_API_KEY", ""),
                "temperature": 0.7,
                "max_tokens": 1024
            },
            "ollama": {
                "enabled": True,
                "model": "llama3.2",
                "base_url": "http://localhost:11434",
                "temperature": 0.7,
                "max_tokens": 1024
            },
            "llama_cpp": {
                "enabled": False,
                "model_path": "",
                "n_ctx": 4096,
                "n_gpu_layers": 0
            },
            "routing": {
                "simple": "ollama",
                "moderate": "ollama",
                "complex": "openai"
            }
        }
    }
    
    save_config(default_config)
    return default_config


# ========== 主入口 ==========

if __name__ == "__main__":
    # 检查配置文件
    config_path = Path("config.yaml")
    if not config_path.exists():
        print("创建默认配置文件...")
        create_default_config()
    
    # 加载配置
    config = load_config()
    server_config = config.get("server", {})
    
    # 启动服务
    uvicorn.run(
        "main:app",
        host=server_config.get("host", "0.0.0.0"),
        port=server_config.get("port", 8000),
        reload=server_config.get("reload", False)
    )
