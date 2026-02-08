"""
SQLite向量存储模块
使用sqlite-vec实现高效的向量检索
支持向量相似度搜索、元数据过滤、混合搜索
"""

import json
import sqlite3
import struct
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple, Any, Union
import numpy as np
import logging

logger = logging.getLogger(__name__)

# ============== 基础数据模型 ==============

@dataclass
class VectorRecord:
    """向量记录"""
    id: str
    vector: List[float]
    metadata: Dict[str, Any]
    text: Optional[str] = None  # 原始文本
    source: str = ""  # 来源
    created_at: Optional[str] = None

@dataclass
class SearchResult:
    """搜索结果"""
    id: str
    score: float  # 相似度分数
    metadata: Dict[str, Any]
    text: Optional[str] = None
    source: str = ""


# ============== 向量相似度计算 ==============

class VectorSimilarity:
    """向量相似度计算工具"""
    
    @staticmethod
    def cosine_similarity(v1: List[float], v2: List[float]) -> float:
        """余弦相似度"""
        v1 = np.array(v1)
        v2 = np.array(v2)
        
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(np.dot(v1, v2) / (norm1 * norm2))
    
    @staticmethod
    def euclidean_distance(v1: List[float], v2: List[float]) -> float:
        """欧氏距离"""
        v1 = np.array(v1)
        v2 = np.array(v2)
        return float(np.linalg.norm(v1 - v2))
    
    @staticmethod
    def dot_product(v1: List[float], v2: List[float]) -> float:
        """点积"""
        return float(np.dot(v1, v2))
    
    @staticmethod
    def normalize(vector: List[float]) -> List[float]:
        """向量归一化"""
        v = np.array(vector)
        norm = np.linalg.norm(v)
        if norm == 0:
            return vector
        return (v / norm).tolist()


# ============== SQLite向量存储 ==============

class SQLiteVectorStore:
    """
    SQLite向量存储
    实现高效的向量存储和相似度搜索
    """
    
    def __init__(self, db_path: str = "vectors.db", 
                 dimension: int = 384,
                 use_fts: bool = True):
        """
        Args:
            db_path: 数据库路径
            dimension: 向量维度
            use_fts: 是否使用全文搜索
        """
        self.db_path = db_path
        self.dimension = dimension
        self.use_fts = use_fts
        self.conn: Optional[sqlite3.Connection] = None
        self.similarity = VectorSimilarity()
        
        self._init_db()
    
    def _init_db(self):
        """初始化数据库"""
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        
        # 启用外键
        self.conn.execute("PRAGMA foreign_keys = ON")
        
        # 创建向量表
        self._create_tables()
        
        logger.info(f"向量数据库初始化完成: {self.db_path}, 维度={self.dimension}")
    
    def _create_tables(self):
        """创建数据表"""
        cursor = self.conn.cursor()
        
        # 主向量表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS vectors (
                id TEXT PRIMARY KEY,
                vector BLOB NOT NULL,  -- 存储为二进制
                text TEXT,
                source TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT  -- JSON格式
            )
        ''')
        
        # 向量索引表（用于快速过滤）
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS vector_index (
                id TEXT PRIMARY KEY,
                magnitude REAL,  -- 向量模长，用于加速余弦相似度计算
                FOREIGN KEY (id) REFERENCES vectors(id) ON DELETE CASCADE
            )
        ''')
        
        # 标签表（用于分类检索）
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS tags (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                vector_id TEXT NOT NULL,
                tag TEXT NOT NULL,
                FOREIGN KEY (vector_id) REFERENCES vectors(id) ON DELETE CASCADE
            )
        ''')
        
        # 创建索引
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_tags_vector_id ON tags(vector_id)
        ''')
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_tags_tag ON tags(tag)
        ''')
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_vectors_source ON vectors(source)
        ''')
        
        # 全文搜索虚拟表（可选）
        if self.use_fts:
            cursor.execute('''
                CREATE VIRTUAL TABLE IF NOT EXISTS vectors_fts USING fts5(
                    id,
                    text,
                    content='vectors',
                    content_rowid='rowid'
                )
            ''')
            
            # 创建触发器保持FTS同步
            cursor.execute('''
                CREATE TRIGGER IF NOT EXISTS vectors_ai AFTER INSERT ON vectors BEGIN
                    INSERT INTO vectors_fts(id, text) VALUES (new.id, new.text);
                END
            ''')
            cursor.execute('''
                CREATE TRIGGER IF NOT EXISTS vectors_ad AFTER DELETE ON vectors BEGIN
                    INSERT INTO vectors_fts(vectors_fts, rowid, id, text) 
                    VALUES ('delete', old.rowid, old.id, old.text);
                END
            ''')
        
        self.conn.commit()
    
    def _vector_to_blob(self, vector: List[float]) -> bytes:
        """将向量转换为二进制存储"""
        return struct.pack(f'{len(vector)}f', *vector)
    
    def _blob_to_vector(self, blob: bytes) -> List[float]:
        """从二进制恢复向量"""
        size = len(blob) // 4  # float32 = 4 bytes
        return list(struct.unpack(f'{size}f', blob))
    
    def add(self, record: VectorRecord) -> bool:
        """
        添加向量记录
        
        Args:
            record: 向量记录
        """
        try:
            cursor = self.conn.cursor()
            
            # 验证向量维度
            if len(record.vector) != self.dimension:
                raise ValueError(f"向量维度不匹配: {len(record.vector)} != {self.dimension}")
            
            # 存储向量
            vector_blob = self._vector_to_blob(record.vector)
            magnitude = np.linalg.norm(record.vector)
            
            cursor.execute('''
                INSERT OR REPLACE INTO vectors 
                (id, vector, text, source, created_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                record.id,
                vector_blob,
                record.text,
                record.source,
                record.created_at,
                json.dumps(record.metadata, ensure_ascii=False)
            ))
            
            # 更新索引
            cursor.execute('''
                INSERT OR REPLACE INTO vector_index (id, magnitude)
                VALUES (?, ?)
            ''', (record.id, float(magnitude)))
            
            # 添加标签
            if 'tags' in record.metadata:
                for tag in record.metadata['tags']:
                    cursor.execute('''
                        INSERT OR IGNORE INTO tags (vector_id, tag)
                        VALUES (?, ?)
                    ''', (record.id, tag))
            
            self.conn.commit()
            return True
            
        except Exception as e:
            logger.error(f"添加向量失败: {e}")
            self.conn.rollback()
            return False
    
    def add_batch(self, records: List[VectorRecord], batch_size: int = 100) -> Tuple[int, int]:
        """
        批量添加向量
        
        Returns:
            (成功数, 失败数)
        """
        success = 0
        failed = 0
        
        for i, record in enumerate(records):
            if self.add(record):
                success += 1
            else:
                failed += 1
            
            if (i + 1) % batch_size == 0:
                logger.info(f"已处理 {i + 1}/{len(records)} 条记录")
        
        return success, failed
    
    def get(self, id: str) -> Optional[VectorRecord]:
        """获取向量记录"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT id, vector, text, source, created_at, metadata
            FROM vectors WHERE id = ?
        ''', (id,))
        
        row = cursor.fetchone()
        if not row:
            return None
        
        return VectorRecord(
            id=row['id'],
            vector=self._blob_to_vector(row['vector']),
            text=row['text'],
            source=row['source'],
            created_at=row['created_at'],
            metadata=json.loads(row['metadata']) if row['metadata'] else {}
        )
    
    def delete(self, id: str) -> bool:
        """删除向量记录"""
        try:
            cursor = self.conn.cursor()
            cursor.execute('DELETE FROM vectors WHERE id = ?', (id,))
            self.conn.commit()
            return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"删除向量失败: {e}")
            return False
    
    def search(self, query_vector: List[float], top_k: int = 10,
               filters: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """
        向量相似度搜索
        
        Args:
            query_vector: 查询向量
            top_k: 返回结果数
            filters: 元数据过滤条件
        """
        if len(query_vector) != self.dimension:
            raise ValueError(f"查询向量维度不匹配: {len(query_vector)} != {self.dimension}")
        
        # 归一化查询向量（用于余弦相似度）
        query_norm = np.linalg.norm(query_vector)
        if query_norm == 0:
            return []
        query_normalized = np.array(query_vector) / query_norm
        
        cursor = self.conn.cursor()
        
        # 构建查询
        where_clause = ""
        params = []
        
        if filters:
            conditions = []
            if 'source' in filters:
                conditions.append("v.source = ?")
                params.append(filters['source'])
            if 'tag' in filters:
                conditions.append("EXISTS (SELECT 1 FROM tags WHERE vector_id = v.id AND tag = ?)")
                params.append(filters['tag'])
            
            if conditions:
                where_clause = "WHERE " + " AND ".join(conditions)
        
        # 获取候选向量（这里获取前100个进行精确计算）
        cursor.execute(f'''
            SELECT v.id, v.vector, v.text, v.source, v.metadata, vi.magnitude
            FROM vectors v
            JOIN vector_index vi ON v.id = vi.id
            {where_clause}
            LIMIT 1000
        ''', params)
        
        results = []
        for row in cursor.fetchall():
            vector = self._blob_to_vector(row['vector'])
            magnitude = row['magnitude']
            
            # 计算余弦相似度
            if magnitude > 0:
                similarity = float(np.dot(query_normalized, vector) / magnitude)
            else:
                similarity = 0.0
            
            results.append((row['id'], similarity, row))
        
        # 排序并取top_k
        results.sort(key=lambda x: x[1], reverse=True)
        top_results = results[:top_k]
        
        return [
            SearchResult(
                id=id,
                score=score,
                metadata=json.loads(row['metadata']) if row['metadata'] else {},
                text=row['text'],
                source=row['source']
            )
            for id, score, row in top_results
        ]
    
    def hybrid_search(self, query_vector: List[float], query_text: str,
                     top_k: int = 10, vector_weight: float = 0.7) -> List[SearchResult]:
        """
        混合搜索：结合向量相似度和文本匹配
        
        Args:
            query_vector: 查询向量
            query_text: 查询文本
            top_k: 返回结果数
            vector_weight: 向量搜索权重（1-vector_weight为文本权重）
        """
        if not self.use_fts:
            logger.warning("全文搜索未启用，回退到纯向量搜索")
            return self.search(query_vector, top_k)
        
        # 向量搜索结果
        vector_results = self.search(query_vector, top_k=top_k*2)
        vector_scores = {r.id: r.score for r in vector_results}
        
        # 全文搜索结果
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT id, rank FROM vectors_fts
            WHERE vectors_fts MATCH ?
            ORDER BY rank
            LIMIT ?
        ''', (query_text, top_k*2))
        
        text_scores = {}
        for row in cursor.fetchall():
            # FTS rank是距离，越小越好，需要反转
            text_scores[row['id']] = 1.0 / (1.0 + abs(row['rank']))
        
        # 融合结果
        all_ids = set(vector_scores.keys()) | set(text_scores.keys())
        combined = []
        
        for id in all_ids:
            v_score = vector_scores.get(id, 0)
            t_score = text_scores.get(id, 0)
            combined_score = vector_weight * v_score + (1 - vector_weight) * t_score
            combined.append((id, combined_score))
        
        combined.sort(key=lambda x: x[1], reverse=True)
        
        # 获取完整记录
        results = []
        for id, score in combined[:top_k]:
            record = self.get(id)
            if record:
                results.append(SearchResult(
                    id=id,
                    score=score,
                    metadata=record.metadata,
                    text=record.text,
                    source=record.source
                ))
        
        return results
    
    def search_by_tag(self, tag: str, top_k: int = 100) -> List[SearchResult]:
        """按标签搜索"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT v.id, v.text, v.source, v.metadata
            FROM vectors v
            JOIN tags t ON v.id = t.vector_id
            WHERE t.tag = ?
            LIMIT ?
        ''', (tag, top_k))
        
        return [
            SearchResult(
                id=row['id'],
                score=1.0,
                metadata=json.loads(row['metadata']) if row['metadata'] else {},
                text=row['text'],
                source=row['source']
            )
            for row in cursor.fetchall()
        ]
    
    def get_stats(self) -> Dict:
        """获取数据库统计信息"""
        cursor = self.conn.cursor()
        
        cursor.execute('SELECT COUNT(*) FROM vectors')
        total_vectors = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(DISTINCT source) FROM vectors')
        unique_sources = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM tags')
        total_tags = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(DISTINCT tag) FROM tags')
        unique_tags = cursor.fetchone()[0]
        
        return {
            'total_vectors': total_vectors,
            'dimension': self.dimension,
            'unique_sources': unique_sources,
            'total_tag_assignments': total_tags,
            'unique_tags': unique_tags,
            'db_path': self.db_path
        }
    
    def close(self):
        """关闭数据库连接"""
        if self.conn:
            self.conn.close()
            self.conn = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# ============== 高级向量索引 ==============

class VectorIndex:
    """
    高级向量索引
    实现IVF（倒排文件）索引加速大规模向量搜索
    """
    
    def __init__(self, vector_store: SQLiteVectorStore, n_clusters: int = 10):
        self.store = vector_store
        self.n_clusters = n_clusters
        self.centroids: List[List[float]] = []
        self.cluster_assignments: Dict[str, int] = {}
    
    def build_index(self):
        """构建索引"""
        # 获取所有向量
        cursor = self.store.conn.cursor()
        cursor.execute('SELECT id, vector FROM vectors')
        
        vectors = []
        ids = []
        for row in cursor.fetchall():
            ids.append(row['id'])
            vectors.append(self.store._blob_to_vector(row['vector']))
        
        if len(vectors) < self.n_clusters:
            logger.warning("向量数量不足，跳过索引构建")
            return
        
        # 简单的k-means聚类
        self.centroids = self._kmeans(vectors, self.n_clusters)
        
        # 分配向量到最近的聚类
        for i, vec in enumerate(vectors):
            closest = self._find_closest_centroid(vec)
            self.cluster_assignments[ids[i]] = closest
        
        logger.info(f"向量索引构建完成: {self.n_clusters} 个聚类")
    
    def _kmeans(self, vectors: List[List[float]], k: int, max_iter: int = 10) -> List[List[float]]:
        """简化版k-means"""
        vectors = np.array(vectors)
        n_samples, n_features = vectors.shape
        
        # 随机初始化中心点
        indices = np.random.choice(n_samples, k, replace=False)
        centroids = vectors[indices].copy()
        
        for _ in range(max_iter):
            # 分配样本
            distances = np.linalg.norm(vectors[:, np.newaxis] - centroids, axis=2)
            labels = np.argmin(distances, axis=1)
            
            # 更新中心点
            new_centroids = np.array([
                vectors[labels == i].mean(axis=0) if np.sum(labels == i) > 0 
                else centroids[i]
                for i in range(k)
            ])
            
            if np.allclose(centroids, new_centroids):
                break
            
            centroids = new_centroids
        
        return centroids.tolist()
    
    def _find_closest_centroid(self, vector: List[float]) -> int:
        """找到最近的聚类中心"""
        min_dist = float('inf')
        closest = 0
        
        for i, centroid in enumerate(self.centroids):
            dist = np.linalg.norm(np.array(vector) - np.array(centroid))
            if dist < min_dist:
                min_dist = dist
                closest = i
        
        return closest
    
    def search(self, query_vector: List[float], top_k: int = 10) -> List[SearchResult]:
        """使用索引加速搜索"""
        if not self.centroids:
            # 回退到线性搜索
            return self.store.search(query_vector, top_k)
        
        # 找到最近的聚类
        closest_cluster = self._find_closest_centroid(query_vector)
        
        # 获取该聚类中的向量ID
        cluster_ids = [
            id for id, cluster in self.cluster_assignments.items()
            if cluster == closest_cluster
        ]
        
        # 在聚类内精确搜索
        results = []
        for id in cluster_ids[:100]:  # 限制搜索数量
            record = self.store.get(id)
            if record:
                score = self.store.similarity.cosine_similarity(
                    query_vector, record.vector
                )
                results.append(SearchResult(
                    id=id,
                    score=score,
                    metadata=record.metadata,
                    text=record.text,
                    source=record.source
                ))
        
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]


# ============== 与主API集成 ==============

class VectorSearchAPI:
    """向量搜索API接口"""
    
    def __init__(self, db_path: str = "vectors.db", dimension: int = 384):
        self.store = SQLiteVectorStore(db_path, dimension)
        self.index: Optional[VectorIndex] = None
    
    def build_index(self, n_clusters: int = 10):
        """构建索引加速搜索"""
        self.index = VectorIndex(self.store, n_clusters)
        self.index.build_index()
    
    def add_document(self, doc_id: str, text: str, embedding: List[float],
                    metadata: Optional[Dict] = None, source: str = "") -> bool:
        """添加文档"""
        record = VectorRecord(
            id=doc_id,
            vector=embedding,
            text=text,
            source=source,
            metadata=metadata or {},
            created_at=datetime.now().isoformat()
        )
        return self.store.add(record)
    
    def search_similar(self, query_embedding: List[float], top_k: int = 10,
                      filters: Optional[Dict] = None) -> List[Dict]:
        """搜索相似向量"""
        if self.index:
            results = self.index.search(query_embedding, top_k)
        else:
            results = self.store.search(query_embedding, top_k, filters)
        
        return [asdict(r) for r in results]
    
    def hybrid_search(self, query_embedding: List[float], query_text: str,
                     top_k: int = 10) -> List[Dict]:
        """混合搜索"""
        results = self.store.hybrid_search(query_embedding, query_text, top_k)
        return [asdict(r) for r in results]
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        return self.store.get_stats()
    
    def close(self):
        """关闭连接"""
        self.store.close()


# ============== 使用示例 ==============

def example_usage():
    """使用示例"""
    
    # 示例1: 基本向量存储
    print("=== 向量存储示例 ===")
    
    with SQLiteVectorStore("example_vectors.db", dimension=128) as store:
        # 添加向量
        records = [
            VectorRecord(
                id=f"doc_{i}",
                vector=[random.random() for _ in range(128)],
                text=f"这是第{i}个文档的内容",
                source="example",
                metadata={"category": "test", "index": i, "tags": ["tag1", "tag2"]}
            )
            for i in range(100)
        ]
        
        success, failed = store.add_batch(records)
        print(f"批量添加: 成功={success}, 失败={failed}")
        
        # 向量搜索
        query = [random.random() for _ in range(128)]
        results = store.search(query, top_k=5)
        
        print("\n向量搜索结果:")
        for r in results:
            print(f"  ID={r.id}, 相似度={r.score:.4f}, 来源={r.source}")
        
        # 标签搜索
        tag_results = store.search_by_tag("tag1", top_k=5)
        print(f"\n标签搜索结果: 找到 {len(tag_results)} 条记录")
        
        # 统计信息
        stats = store.get_stats()
        print(f"\n统计信息: {stats}")
    
    # 示例2: 混合搜索
    print("\n=== 混合搜索示例 ===")
    
    with SQLiteVectorStore("example_hybrid.db", dimension=128, use_fts=True) as store:
        # 添加带文本的向量
        documents = [
            VectorRecord(
                id="doc_ai",
                vector=[random.random() for _ in range(128)],
                text="人工智能和机器学习正在改变世界",
                source="tech",
                metadata={"topic": "AI"}
            ),
            VectorRecord(
                id="doc_ml",
                vector=[random.random() for _ in range(128)],
                text="机器学习是人工智能的一个分支",
                source="tech",
                metadata={"topic": "ML"}
            ),
            VectorRecord(
                id="doc_cooking",
                vector=[random.random() for _ in range(128)],
                text="烹饪是一门艺术，需要技巧和创意",
                source="lifestyle",
                metadata={"topic": "cooking"}
            ),
        ]
        
        for doc in documents:
            store.add(doc)
        
        # 混合搜索
        query_vector = [random.random() for _ in range(128)]
        hybrid_results = store.hybrid_search(
            query_vector, 
            query_text="人工智能",
            top_k=3
        )
        
        print("混合搜索结果:")
        for r in hybrid_results:
            print(f"  ID={r.id}, 分数={r.score:.4f}")
            print(f"    文本: {r.text}")
    
    # 示例3: API集成
    print("\n=== API集成示例 ===")
    
    api = VectorSearchAPI("example_api.db", dimension=384)
    
    # 添加文档
    for i in range(10):
        api.add_document(
            doc_id=f"api_doc_{i}",
            text=f"API文档示例内容 {i}",
            embedding=[random.random() for _ in range(384)],
            metadata={"version": "1.0", "author": "system"},
            source="api"
        )
    
    # 搜索
    query_emb = [random.random() for _ in range(384)]
    search_results = api.search_similar(query_emb, top_k=3)
    
    print(f"API搜索结果: {len(search_results)} 条")
    for r in search_results:
        print(f"  {r['id']}: 分数={r['score']:.4f}")
    
    print(f"\n统计: {api.get_stats()}")
    
    api.close()
    
    # 清理示例数据库
    import os
    for db in ["example_vectors.db", "example_hybrid.db", "example_api.db"]:
        if os.path.exists(db):
            os.remove(db)


if __name__ == "__main__":
    import random
    example_usage()
