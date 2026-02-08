"""
Embedding服务 - 基于Sentence-Transformers
支持多种模型和批量生成
"""
import json
import pickle
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Union, Any, Tuple
from dataclasses import dataclass, field
import numpy as np


@dataclass
class EmbeddingResult:
    """Embedding结果"""
    text: str
    embedding: np.ndarray
    model: str
    dimension: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "text": self.text[:100] + "..." if len(self.text) > 100 else self.text,
            "embedding_shape": self.embedding.shape,
            "model": self.model,
            "dimension": self.dimension,
            "metadata": self.metadata
        }


class EmbeddingCache:
    """Embedding缓存管理"""
    
    def __init__(self, cache_dir: str = "embedding_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.memory_cache: Dict[str, np.ndarray] = {}
        self.max_memory_items = 10000
    
    def _get_cache_key(self, text: str, model: str) -> str:
        """生成缓存键"""
        content = f"{text}:{model}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _get_cache_path(self, key: str) -> Path:
        """获取缓存文件路径"""
        return self.cache_dir / f"{key}.pkl"
    
    def get(self, text: str, model: str) -> Optional[np.ndarray]:
        """获取缓存的embedding"""
        key = self._get_cache_key(text, model)
        
        # 先检查内存缓存
        if key in self.memory_cache:
            return self.memory_cache[key]
        
        # 检查磁盘缓存
        cache_path = self._get_cache_path(key)
        if cache_path.exists():
            with open(cache_path, 'rb') as f:
                embedding = pickle.load(f)
                # 加入内存缓存
                self._add_to_memory(key, embedding)
                return embedding
        
        return None
    
    def set(self, text: str, model: str, embedding: np.ndarray):
        """设置缓存"""
        key = self._get_cache_key(text, model)
        
        # 保存到磁盘
        cache_path = self._get_cache_path(key)
        with open(cache_path, 'wb') as f:
            pickle.dump(embedding, f)
        
        # 添加到内存缓存
        self._add_to_memory(key, embedding)
    
    def _add_to_memory(self, key: str, embedding: np.ndarray):
        """添加到内存缓存（LRU）"""
        if len(self.memory_cache) >= self.max_memory_items:
            # 移除最早的项
            oldest_key = next(iter(self.memory_cache))
            del self.memory_cache[oldest_key]
        
        self.memory_cache[key] = embedding
    
    def clear(self):
        """清空缓存"""
        self.memory_cache.clear()
        for f in self.cache_dir.glob("*.pkl"):
            f.unlink()
    
    def get_stats(self) -> Dict:
        """获取缓存统计"""
        disk_count = len(list(self.cache_dir.glob("*.pkl")))
        return {
            "memory_items": len(self.memory_cache),
            "disk_items": disk_count,
            "cache_dir": str(self.cache_dir)
        }


class EmbeddingService:
    """
    Embedding服务主类
    支持多种sentence-transformer模型
    """
    
    # 预定义的模型配置
    MODEL_CONFIGS = {
        "all-MiniLM-L6-v2": {
            "name": "sentence-transformers/all-MiniLM-L6-v2",
            "dimension": 384,
            "description": "轻量级模型，适合快速推理",
            "max_seq_length": 256,
            "language": "multilingual"
        },
        "all-MiniLM-L12-v2": {
            "name": "sentence-transformers/all-MiniLM-L12-v2",
            "dimension": 384,
            "description": "稍大的MiniLM模型，性能更好",
            "max_seq_length": 256,
            "language": "multilingual"
        },
        "all-mpnet-base-v2": {
            "name": "sentence-transformers/all-mpnet-base-v2",
            "dimension": 768,
            "description": "当前最佳通用模型",
            "max_seq_length": 384,
            "language": "multilingual"
        },
        "bge-large-en": {
            "name": "BAAI/bge-large-en",
            "dimension": 1024,
            "description": "BGE英文大模型，检索效果优秀",
            "max_seq_length": 512,
            "language": "en"
        },
        "bge-large-zh": {
            "name": "BAAI/bge-large-zh",
            "dimension": 1024,
            "description": "BGE中文大模型",
            "max_seq_length": 512,
            "language": "zh"
        },
        "bge-m3": {
            "name": "BAAI/bge-m3",
            "dimension": 1024,
            "description": "BGE多语言大模型",
            "max_seq_length": 8192,
            "language": "multilingual"
        },
        "e5-large-v2": {
            "name": "intfloat/e5-large-v2",
            "dimension": 1024,
            "description": "E5英文模型",
            "max_seq_length": 512,
            "language": "en"
        },
        "paraphrase-multilingual-mpnet-base-v2": {
            "name": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
            "dimension": 768,
            "description": "多语言释义检测模型",
            "max_seq_length": 128,
            "language": "multilingual"
        },
    }
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.default_model = self.config.get("default_model", "all-MiniLM-L6-v2")
        self.device = self.config.get("device", "auto")
        self.use_cache = self.config.get("use_cache", True)
        self.normalize_embeddings = self.config.get("normalize", True)
        
        # 模型实例缓存
        self._models: Dict[str, Any] = {}
        
        # Embedding缓存
        self.cache = EmbeddingCache(
            cache_dir=self.config.get("cache_dir", "embedding_cache")
        ) if self.use_cache else None
        
        # 加载默认模型
        if self.config.get("auto_load", True):
            self.load_model(self.default_model)
    
    def list_models(self) -> List[Dict]:
        """列出可用模型"""
        return [
            {
                "id": key,
                **config
            }
            for key, config in self.MODEL_CONFIGS.items()
        ]
    
    def load_model(self, model_id: str) -> bool:
        """
        加载指定的embedding模型
        
        Args:
            model_id: 模型ID
        
        Returns:
            是否成功加载
        """
        if model_id in self._models:
            return True
        
        try:
            from sentence_transformers import SentenceTransformer
            
            if model_id not in self.MODEL_CONFIGS:
                # 尝试直接使用模型名称
                model_name = model_id
            else:
                model_name = self.MODEL_CONFIGS[model_id]["name"]
            
            print(f"Loading embedding model: {model_name}")
            
            # 加载模型
            model = SentenceTransformer(model_name, device=self.device)
            
            self._models[model_id] = {
                "model": model,
                "config": self.MODEL_CONFIGS.get(model_id, {
                    "name": model_name,
                    "dimension": model.get_sentence_embedding_dimension(),
                    "max_seq_length": model.max_seq_length
                })
            }
            
            print(f"Model loaded successfully. Dimension: {self._models[model_id]['config']['dimension']}")
            return True
            
        except Exception as e:
            print(f"Failed to load model {model_id}: {e}")
            return False
    
    def encode(
        self,
        texts: Union[str, List[str]],
        model_id: Optional[str] = None,
        batch_size: int = 32,
        show_progress: bool = False,
        use_cache: Optional[bool] = None
    ) -> Union[EmbeddingResult, List[EmbeddingResult]]:
        """
        编码文本为embedding向量
        
        Args:
            texts: 输入文本或文本列表
            model_id: 模型ID，默认使用default_model
            batch_size: 批处理大小
            show_progress: 是否显示进度条
            use_cache: 是否使用缓存
        
        Returns:
            Embedding结果或结果列表
        """
        if use_cache is None:
            use_cache = self.use_cache
        
        model_id = model_id or self.default_model
        
        # 确保模型已加载
        if model_id not in self._models:
            if not self.load_model(model_id):
                raise ValueError(f"Failed to load model: {model_id}")
        
        # 统一处理为列表
        is_single = isinstance(texts, str)
        text_list = [texts] if is_single else texts
        
        # 检查缓存
        if use_cache and self.cache:
            cached_results = []
            texts_to_encode = []
            indices_to_encode = []
            
            for i, text in enumerate(text_list):
                cached = self.cache.get(text, model_id)
                if cached is not None:
                    cached_results.append((i, cached))
                else:
                    texts_to_encode.append(text)
                    indices_to_encode.append(i)
        else:
            texts_to_encode = text_list
            indices_to_encode = list(range(len(text_list)))
            cached_results = []
        
        # 编码未缓存的文本
        if texts_to_encode:
            model_data = self._models[model_id]
            model = model_data["model"]
            config = model_data["config"]
            
            embeddings = model.encode(
                texts_to_encode,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                normalize_embeddings=self.normalize_embeddings
            )
            
            # 保存到缓存
            if use_cache and self.cache:
                for text, emb in zip(texts_to_encode, embeddings):
                    self.cache.set(text, model_id, emb)
        else:
            embeddings = np.array([])
        
        # 合并结果
        all_embeddings = [None] * len(text_list)
        
        # 填入缓存结果
        for idx, emb in cached_results:
            all_embeddings[idx] = emb
        
        # 填入新编码结果
        for idx, emb in zip(indices_to_encode, embeddings):
            all_embeddings[idx] = emb
        
        # 创建结果对象
        results = []
        model_data = self._models[model_id]
        for text, emb in zip(text_list, all_embeddings):
            results.append(EmbeddingResult(
                text=text,
                embedding=emb,
                model=model_id,
                dimension=len(emb),
                metadata={"normalized": self.normalize_embeddings}
            ))
        
        return results[0] if is_single else results
    
    def encode_batch(
        self,
        texts: List[str],
        model_id: Optional[str] = None,
        batch_size: int = 32
    ) -> np.ndarray:
        """
        批量编码，返回numpy数组
        
        Args:
            texts: 文本列表
            model_id: 模型ID
            batch_size: 批处理大小
        
        Returns:
            embeddings数组 (N, D)
        """
        results = self.encode(texts, model_id, batch_size)
        return np.array([r.embedding for r in results])
    
    def similarity(
        self,
        text1: Union[str, np.ndarray],
        text2: Union[str, np.ndarray],
        model_id: Optional[str] = None
    ) -> float:
        """
        计算两个文本的相似度
        
        Args:
            text1: 文本1或embedding
            text2: 文本2或embedding
            model_id: 模型ID
        
        Returns:
            余弦相似度 (-1 to 1)
        """
        # 获取embeddings
        if isinstance(text1, str):
            emb1 = self.encode(text1, model_id).embedding
        else:
            emb1 = text1
        
        if isinstance(text2, str):
            emb2 = self.encode(text2, model_id).embedding
        else:
            emb2 = text2
        
        # 计算余弦相似度
        dot_product = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot_product / (norm1 * norm2))
    
    def find_similar(
        self,
        query: str,
        candidates: List[str],
        top_k: int = 5,
        model_id: Optional[str] = None,
        threshold: float = 0.0
    ) -> List[Dict]:
        """
        在候选文本中查找与查询最相似的
        
        Args:
            query: 查询文本
            candidates: 候选文本列表
            top_k: 返回前k个结果
            model_id: 模型ID
            threshold: 相似度阈值
        
        Returns:
            相似结果列表
        """
        # 编码查询和候选
        query_emb = self.encode(query, model_id).embedding
        candidate_embs = self.encode_batch(candidates, model_id)
        
        # 计算相似度
        similarities = np.dot(candidate_embs, query_emb)
        
        # 排序并过滤
        indices = np.argsort(similarities)[::-1]
        results = []
        
        for idx in indices[:top_k]:
            sim = float(similarities[idx])
            if sim >= threshold:
                results.append({
                    "text": candidates[idx],
                    "similarity": sim,
                    "rank": len(results) + 1
                })
        
        return results
    
    def semantic_search(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        text_key: str = "text",
        top_k: int = 5,
        model_id: Optional[str] = None
    ) -> List[Dict]:
        """
        语义搜索
        
        Args:
            query: 查询
            documents: 文档列表
            text_key: 文本字段名
            top_k: 返回数量
            model_id: 模型ID
        
        Returns:
            搜索结果
        """
        texts = [doc.get(text_key, "") for doc in documents]
        similarities = self.find_similar(query, texts, top_k=len(texts), model_id=model_id)
        
        # 合并文档信息
        results = []
        for sim_result in similarities[:top_k]:
            # 找到对应的文档
            doc_idx = texts.index(sim_result["text"])
            result = {
                **documents[doc_idx],
                "similarity": sim_result["similarity"],
                "rank": len(results) + 1
            }
            results.append(result)
        
        return results
    
    def clustering(
        self,
        texts: List[str],
        n_clusters: int = 5,
        model_id: Optional[str] = None
    ) -> List[Dict]:
        """
        对文本进行聚类
        
        Args:
            texts: 文本列表
            n_clusters: 聚类数量
            model_id: 模型ID
        
        Returns:
            聚类结果
        """
        from sklearn.cluster import KMeans
        
        # 获取embeddings
        embeddings = self.encode_batch(texts, model_id)
        
        # 聚类
        n_clusters = min(n_clusters, len(texts))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(embeddings)
        
        # 组织结果
        clusters = [[] for _ in range(n_clusters)]
        for i, (text, label) in enumerate(zip(texts, labels)):
            clusters[label].append({
                "text": text,
                "index": i
            })
        
        return [
            {
                "cluster_id": i,
                "size": len(cluster),
                "items": cluster
            }
            for i, cluster in enumerate(clusters)
        ]
    
    def get_model_info(self, model_id: Optional[str] = None) -> Dict:
        """获取模型信息"""
        model_id = model_id or self.default_model
        
        if model_id in self._models:
            return {
                "loaded": True,
                **self._models[model_id]["config"]
            }
        elif model_id in self.MODEL_CONFIGS:
            return {
                "loaded": False,
                **self.MODEL_CONFIGS[model_id]
            }
        else:
            return {"loaded": False, "name": model_id}
    
    def get_cache_stats(self) -> Optional[Dict]:
        """获取缓存统计"""
        return self.cache.get_stats() if self.cache else None
    
    def clear_cache(self):
        """清空缓存"""
        if self.cache:
            self.cache.clear()


class MultiModelEmbeddingService:
    """
    多模型Embedding服务
    支持同时使用多个模型，自动选择最合适的
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.services: Dict[str, EmbeddingService] = {}
        
        # 语言到模型的映射
        self.language_models = {
            "zh": self.config.get("chinese_model", "bge-large-zh"),
            "en": self.config.get("english_model", "bge-large-en"),
            "multilingual": self.config.get("multilingual_model", "bge-m3"),
        }
        
        # 默认服务
        self.default_service = EmbeddingService({
            "default_model": self.config.get("default_model", "all-MiniLM-L6-v2"),
            "auto_load": True
        })
    
    def _detect_language(self, text: str) -> str:
        """检测文本语言"""
        import re
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        total_chars = len(text.replace(" ", ""))
        if total_chars > 0 and chinese_chars / total_chars > 0.5:
            return "zh"
        return "en"
    
    def encode(
        self,
        texts: Union[str, List[str]],
        language: Optional[str] = None,
        **kwargs
    ) -> Union[EmbeddingResult, List[EmbeddingResult]]:
        """
        智能编码，自动选择模型
        
        Args:
            texts: 输入文本
            language: 指定语言，自动检测
        """
        is_single = isinstance(texts, str)
        text = texts if is_single else texts[0]
        
        if language is None:
            language = self._detect_language(text)
        
        # 选择合适的模型
        model_id = self.language_models.get(language, self.language_models["multilingual"])
        
        # 确保服务存在
        if model_id not in self.services:
            self.services[model_id] = EmbeddingService({
                "default_model": model_id,
                "auto_load": False
            })
            self.services[model_id].load_model(model_id)
        
        return self.services[model_id].encode(texts, **kwargs)


# 便捷函数
def create_embedding_service(config: Optional[Dict] = None) -> EmbeddingService:
    """创建Embedding服务"""
    return EmbeddingService(config)


def quick_encode(text: Union[str, List[str]], model: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """快速编码"""
    service = EmbeddingService({"default_model": model, "auto_load": True})
    result = service.encode(text)
    
    if isinstance(result, list):
        return np.array([r.embedding for r in result])
    return result.embedding


def compute_similarity(text1: str, text2: str, model: str = "all-MiniLM-L6-v2") -> float:
    """快速计算相似度"""
    service = EmbeddingService({"default_model": model, "auto_load": True})
    return service.similarity(text1, text2)


if __name__ == "__main__":
    # 测试
    service = EmbeddingService()
    
    # 列出可用模型
    print("可用模型:")
    for model in service.list_models():
        print(f"  - {model['id']}: {model['description']} (dim={model['dimension']})")
    
    # 单文本编码
    print("\n单文本编码:")
    result = service.encode("这是一个测试句子")
    print(f"维度: {result.dimension}")
    print(f"前5个值: {result.embedding[:5]}")
    
    # 批量编码
    print("\n批量编码:")
    texts = ["机器学习很有趣", "深度学习是AI的子集", "自然语言处理"]
    results = service.encode(texts)
    print(f"编码了 {len(results)} 个文本")
    
    # 相似度计算
    print("\n相似度计算:")
    sim = service.similarity("机器学习", "深度学习")
    print(f"'机器学习' vs '深度学习': {sim:.4f}")
    
    sim2 = service.similarity("机器学习", "苹果香蕉")
    print(f"'机器学习' vs '苹果香蕉': {sim2:.4f}")
    
    # 语义搜索
    print("\n语义搜索:")
    docs = [
        {"id": 1, "text": "Python是一种编程语言"},
        {"id": 2, "text": "JavaScript用于网页开发"},
        {"id": 3, "text": "机器学习需要大量数据"},
        {"id": 4, "text": "深度学习使用神经网络"},
    ]
    results = service.semantic_search("AI技术", docs, top_k=2)
    for r in results:
        print(f"  {r['rank']}. {r['text']} (相似度: {r['similarity']:.4f})")
    
    # 聚类
    print("\n聚类:")
    cluster_texts = [
        "苹果是一种水果",
        "香蕉富含维生素",
        "机器学习是AI的分支",
        "深度学习需要GPU",
        "橙子很甜",
        "神经网络很复杂"
    ]
    clusters = service.clustering(cluster_texts, n_clusters=2)
    for c in clusters:
        print(f"  Cluster {c['cluster_id']} ({c['size']} items):")
        for item in c['items']:
            print(f"    - {item['text']}")
    
    # 缓存统计
    print("\n缓存统计:")
    print(service.get_cache_stats())
