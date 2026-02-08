"""
推荐系统增强模块 V2
实现协同过滤、内容推荐、混合推荐和A/B测试框架
"""

import json
import hashlib
import random
from abc import ABC, abstractmethod
from collections import defaultdict, Counter
from dataclasses import dataclass, asdict, field
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any, Set, Callable
import math
import logging

logger = logging.getLogger(__name__)

# ============== 基础数据模型 ==============

@dataclass
class User:
    """用户模型"""
    id: str
    features: Dict[str, Any] = field(default_factory=dict)
    history: List[str] = field(default_factory=list)  # 交互过的物品ID
    ratings: Dict[str, float] = field(default_factory=dict)  # 物品评分
    demographic: Dict[str, Any] = field(default_factory=dict)  # 人口统计信息
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class Item:
    """物品模型"""
    id: str
    title: str
    features: Dict[str, Any] = field(default_factory=dict)  # 内容特征
    category: str = ""
    tags: List[str] = field(default_factory=list)
    popularity: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class Interaction:
    """用户-物品交互"""
    user_id: str
    item_id: str
    rating: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)
    context: Dict[str, Any] = field(default_factory=dict)  # 交互上下文

@dataclass
class Recommendation:
    """推荐结果"""
    item_id: str
    score: float
    reason: str = ""  # 推荐理由
    method: str = ""  # 推荐方法
    confidence: float = 0.0

@dataclass
class ABTestVariant:
    """A/B测试变体"""
    id: str
    name: str
    algorithm: str
    config: Dict[str, Any] = field(default_factory=dict)
    traffic_percentage: float = 0.5


# ============== 1. 协同过滤推荐 ==============

class CollaborativeFiltering:
    """
    协同过滤推荐
    基于用户的协同过滤 + 基于物品的协同过滤
    """
    
    def __init__(self, similarity_metric: str = "cosine"):
        self.similarity_metric = similarity_metric
        self.user_item_matrix: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.item_user_matrix: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.user_similarities: Dict[str, Dict[str, float]] = {}
        self.item_similarities: Dict[str, Dict[str, float]] = {}
    
    def fit(self, interactions: List[Interaction]):
        """训练模型"""
        # 构建用户-物品矩阵
        for inter in interactions:
            rating = inter.rating if inter.rating is not None else 1.0
            self.user_item_matrix[inter.user_id][inter.item_id] = rating
            self.item_user_matrix[inter.item_id][inter.user_id] = rating
        
        logger.info(f"协同过滤: {len(self.user_item_matrix)} 用户, "
                   f"{len(self.item_user_matrix)} 物品")
    
    def user_based_recommend(self, user_id: str, top_k: int = 10,
                             n_similar_users: int = 20) -> List[Recommendation]:
        """
        基于用户的协同过滤推荐
        
        找到相似用户喜欢的物品
        """
        if user_id not in self.user_item_matrix:
            return []
        
        # 找到相似用户
        similar_users = self._get_similar_users(user_id, n_similar_users)
        
        # 收集推荐候选
        candidates: Dict[str, float] = defaultdict(float)
        user_items = set(self.user_item_matrix[user_id].keys())
        
        for similar_user, similarity in similar_users:
            for item_id, rating in self.user_item_matrix[similar_user].items():
                if item_id not in user_items:  # 只推荐用户没交互过的
                    candidates[item_id] += similarity * rating
        
        # 排序并返回
        sorted_candidates = sorted(
            candidates.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:top_k]
        
        return [
            Recommendation(
                item_id=item_id,
                score=score,
                reason=f"与你相似的用户也喜欢",
                method="user_based_cf",
                confidence=min(score / 5, 1.0)
            )
            for item_id, score in sorted_candidates
        ]
    
    def item_based_recommend(self, user_id: str, top_k: int = 10) -> List[Recommendation]:
        """
        基于物品的协同过滤推荐
        
        找到与用户已喜欢物品相似的物品
        """
        if user_id not in self.user_item_matrix:
            return []
        
        user_items = self.user_item_matrix[user_id]
        candidates: Dict[str, List[float]] = defaultdict(list)
        
        for user_item, user_rating in user_items.items():
            similar_items = self._get_similar_items(user_item, 10)
            
            for similar_item, similarity in similar_items:
                if similar_item not in user_items:
                    candidates[similar_item].append(similarity * user_rating)
        
        # 计算加权平均
        scores = {
            item_id: sum(scores) / len(scores)
            for item_id, scores in candidates.items()
        }
        
        sorted_candidates = sorted(
            scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
        
        return [
            Recommendation(
                item_id=item_id,
                score=score,
                reason=f"与你喜欢的物品相似",
                method="item_based_cf",
                confidence=min(score / 5, 1.0)
            )
            for item_id, score in sorted_candidates
        ]
    
    def _get_similar_users(self, user_id: str, n: int) -> List[Tuple[str, float]]:
        """获取相似用户"""
        if user_id not in self.user_similarities:
            self.user_similarities[user_id] = {}
            
            user_vector = self.user_item_matrix[user_id]
            
            for other_id, other_vector in self.user_item_matrix.items():
                if other_id != user_id:
                    similarity = self._calculate_similarity(user_vector, other_vector)
                    if similarity > 0:
                        self.user_similarities[user_id][other_id] = similarity
        
        similarities = self.user_similarities[user_id]
        return sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:n]
    
    def _get_similar_items(self, item_id: str, n: int) -> List[Tuple[str, float]]:
        """获取相似物品"""
        if item_id not in self.item_similarities:
            self.item_similarities[item_id] = {}
            
            item_vector = self.item_user_matrix[item_id]
            
            for other_id, other_vector in self.item_user_matrix.items():
                if other_id != item_id:
                    similarity = self._calculate_similarity(item_vector, other_vector)
                    if similarity > 0:
                        self.item_similarities[item_id][other_id] = similarity
        
        similarities = self.item_similarities[item_id]
        return sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:n]
    
    def _calculate_similarity(self, vec1: Dict[str, float], 
                              vec2: Dict[str, float]) -> float:
        """计算两个向量的相似度"""
        if self.similarity_metric == "cosine":
            return self._cosine_similarity(vec1, vec2)
        elif self.similarity_metric == "pearson":
            return self._pearson_correlation(vec1, vec2)
        else:
            return self._cosine_similarity(vec1, vec2)
    
    def _cosine_similarity(self, vec1: Dict[str, float], 
                           vec2: Dict[str, float]) -> float:
        """余弦相似度"""
        common_keys = set(vec1.keys()) & set(vec2.keys())
        if not common_keys:
            return 0.0
        
        dot_product = sum(vec1[k] * vec2[k] for k in common_keys)
        norm1 = math.sqrt(sum(v ** 2 for v in vec1.values()))
        norm2 = math.sqrt(sum(v ** 2 for v in vec2.values()))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def _pearson_correlation(self, vec1: Dict[str, float],
                             vec2: Dict[str, float]) -> float:
        """皮尔逊相关系数"""
        common_keys = set(vec1.keys()) & set(vec2.keys())
        if len(common_keys) < 2:
            return 0.0
        
        mean1 = sum(vec1[k] for k in common_keys) / len(common_keys)
        mean2 = sum(vec2[k] for k in common_keys) / len(common_keys)
        
        numerator = sum((vec1[k] - mean1) * (vec2[k] - mean2) for k in common_keys)
        denominator1 = math.sqrt(sum((vec1[k] - mean1) ** 2 for k in common_keys))
        denominator2 = math.sqrt(sum((vec2[k] - mean2) ** 2 for k in common_keys))
        
        if denominator1 == 0 or denominator2 == 0:
            return 0.0
        
        return numerator / (denominator1 * denominator2)


# ============== 2. 内容推荐 ==============

class ContentBasedRecommender:
    """
    基于内容的推荐
    根据用户历史偏好和物品特征匹配
    """
    
    def __init__(self):
        self.user_profiles: Dict[str, Dict[str, float]] = {}
        self.items: Dict[str, Item] = {}
        self.item_features_matrix: Dict[str, Dict[str, float]] = {}
    
    def fit(self, users: List[User], items: List[Item], 
            interactions: List[Interaction]):
        """训练模型"""
        # 存储物品信息
        for item in items:
            self.items[item.id] = item
            self.item_features_matrix[item.id] = self._extract_features(item)
        
        # 构建用户画像
        user_item_ratings = defaultdict(dict)
        for inter in interactions:
            rating = inter.rating if inter.rating is not None else 1.0
            user_item_ratings[inter.user_id][inter.item_id] = rating
        
        for user in users:
            self.user_profiles[user.id] = self._build_user_profile(
                user, user_item_ratings.get(user.id, {})
            )
        
        logger.info(f"内容推荐: {len(self.user_profiles)} 用户画像已构建")
    
    def _extract_features(self, item: Item) -> Dict[str, float]:
        """从物品中提取特征向量"""
        features = {}
        
        # 类别特征
        if item.category:
            features[f"cat_{item.category}"] = 1.0
        
        # 标签特征
        for tag in item.tags:
            features[f"tag_{tag}"] = 1.0
        
        # 其他特征
        features.update(item.features)
        
        # 归一化
        total = math.sqrt(sum(v ** 2 for v in features.values()))
        if total > 0:
            features = {k: v / total for k, v in features.items()}
        
        return features
    
    def _build_user_profile(self, user: User, 
                           ratings: Dict[str, float]) -> Dict[str, float]:
        """构建用户画像（加权特征平均）"""
        profile = defaultdict(float)
        total_weight = 0
        
        for item_id, rating in ratings.items():
            if item_id in self.item_features_matrix:
                item_features = self.item_features_matrix[item_id]
                weight = rating
                
                for feature, value in item_features.items():
                    profile[feature] += value * weight
                
                total_weight += weight
        
        # 归一化
        if total_weight > 0:
            profile = {k: v / total_weight for k, v in profile.items()}
        
        return dict(profile)
    
    def recommend(self, user_id: str, top_k: int = 10,
                 exclude_seen: bool = True) -> List[Recommendation]:
        """为用户生成推荐"""
        if user_id not in self.user_profiles:
            return []
        
        user_profile = self.user_profiles[user_id]
        
        # 计算与所有物品的相似度
        scores = []
        for item_id, item_features in self.item_features_matrix.items():
            score = self._calculate_similarity(user_profile, item_features)
            scores.append((item_id, score))
        
        # 排序
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # 返回top_k
        recommendations = []
        for item_id, score in scores[:top_k]:
            item = self.items.get(item_id)
            reason = self._generate_reason(user_profile, item)
            
            recommendations.append(Recommendation(
                item_id=item_id,
                score=score,
                reason=reason,
                method="content_based",
                confidence=min(score * 2, 1.0)  # 放大分数作为置信度
            ))
        
        return recommendations
    
    def _calculate_similarity(self, profile: Dict[str, float],
                             features: Dict[str, float]) -> float:
        """计算用户画像和物品特征的相似度"""
        common_features = set(profile.keys()) & set(features.keys())
        if not common_features:
            return 0.0
        
        dot_product = sum(profile[f] * features[f] for f in common_features)
        
        profile_norm = math.sqrt(sum(v ** 2 for v in profile.values()))
        features_norm = math.sqrt(sum(v ** 2 for v in features.values()))
        
        if profile_norm == 0 or features_norm == 0:
            return 0.0
        
        return dot_product / (profile_norm * features_norm)
    
    def _generate_reason(self, profile: Dict[str, float], item: Item) -> str:
        """生成推荐理由"""
        if not item:
            return "基于你的兴趣"
        
        matching_features = []
        
        if item.category:
            cat_feature = f"cat_{item.category}"
            if cat_feature in profile and profile[cat_feature] > 0.3:
                matching_features.append(item.category)
        
        for tag in item.tags[:3]:
            tag_feature = f"tag_{tag}"
            if tag_feature in profile and profile[tag_feature] > 0.2:
                matching_features.append(tag)
        
        if matching_features:
            return f"因为你对{', '.join(matching_features[:2])}感兴趣"
        
        return "符合你的偏好"
    
    def explain_recommendation(self, user_id: str, item_id: str) -> Dict:
        """解释特定推荐的原因"""
        if user_id not in self.user_profiles or item_id not in self.item_features_matrix:
            return {}
        
        profile = self.user_profiles[user_id]
        features = self.item_features_matrix[item_id]
        item = self.items.get(item_id)
        
        matching = []
        for feature, value in features.items():
            if feature in profile and profile[feature] > 0.1:
                matching.append({
                    'feature': feature,
                    'user_preference': profile[feature],
                    'item_strength': value,
                    'contribution': profile[feature] * value
                })
        
        matching.sort(key=lambda x: x['contribution'], reverse=True)
        
        return {
            'item_id': item_id,
            'item_title': item.title if item else "",
            'overall_score': self._calculate_similarity(profile, features),
            'top_matching_features': matching[:5]
        }


# ============== 3. 混合推荐策略 ==============

class HybridRecommender:
    """
    混合推荐器
    整合多种推荐策略
    """
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """
        Args:
            weights: 各推荐算法的权重
        """
        self.cf = CollaborativeFiltering()
        self.content = ContentBasedRecommender()
        
        # 默认权重
        self.weights = weights or {
            'user_cf': 0.3,
            'item_cf': 0.3,
            'content': 0.4
        }
        
        self.popular_items: List[str] = []
    
    def fit(self, users: List[User], items: List[Item],
            interactions: List[Interaction]):
        """训练所有子模型"""
        self.cf.fit(interactions)
        self.content.fit(users, items, interactions)
        
        # 计算热门物品（用于冷启动）
        item_counts = Counter(inter.item_id for inter in interactions)
        self.popular_items = [item for item, _ in item_counts.most_common(100)]
        
        logger.info("混合推荐模型训练完成")
    
    def recommend(self, user_id: str, top_k: int = 10,
                 context: Dict[str, Any] = None) -> List[Recommendation]:
        """
        生成混合推荐
        
        Args:
            context: 上下文信息，如时间、地点等
        """
        all_candidates: Dict[str, List[Recommendation]] = {}
        
        # 1. 用户协同过滤
        user_cf_recs = self.cf.user_based_recommend(user_id, top_k=top_k*2)
        all_candidates['user_cf'] = user_cf_recs
        
        # 2. 物品协同过滤
        item_cf_recs = self.cf.item_based_recommend(user_id, top_k=top_k*2)
        all_candidates['item_cf'] = item_cf_recs
        
        # 3. 内容推荐
        content_recs = self.content.recommend(user_id, top_k=top_k*2)
        all_candidates['content'] = content_recs
        
        # 融合结果
        merged = self._merge_recommendations(all_candidates, top_k)
        
        # 应用上下文过滤
        if context:
            merged = self._apply_context(merged, context)
        
        return merged
    
    def _merge_recommendations(self, candidates: Dict[str, List[Recommendation]],
                               top_k: int) -> List[Recommendation]:
        """融合多源推荐结果"""
        # 收集所有候选物品
        item_scores: Dict[str, Dict] = defaultdict(lambda: {
            'scores': {},
            'reasons': [],
            'methods': set()
        })
        
        for method, recs in candidates.items():
            weight = self.weights.get(method, 0.33)
            for rec in recs:
                item_scores[rec.item_id]['scores'][method] = rec.score * weight
                item_scores[rec.item_id]['reasons'].append(rec.reason)
                item_scores[rec.item_id]['methods'].add(method)
        
        # 计算加权分数
        final_scores = []
        for item_id, data in item_scores.items():
            total_score = sum(data['scores'].values())
            methods = list(data['methods'])
            
            final_scores.append(Recommendation(
                item_id=item_id,
                score=total_score,
                reason=self._combine_reasons(data['reasons']),
                method=f"hybrid({','.join(methods)})",
                confidence=min(total_score / 3, 1.0)
            ))
        
        # 排序并返回
        final_scores.sort(key=lambda x: x.score, reverse=True)
        return final_scores[:top_k]
    
    def _combine_reasons(self, reasons: List[str]) -> str:
        """组合多个推荐理由"""
        unique_reasons = list(set(reasons))
        if len(unique_reasons) <= 2:
            return "; ".join(unique_reasons)
        return f"{unique_reasons[0]}等{len(unique_reasons)}个原因"
    
    def _apply_context(self, recommendations: List[Recommendation],
                      context: Dict[str, Any]) -> List[Recommendation]:
        """应用上下文调整推荐"""
        time_of_day = context.get('time_of_day')
        location = context.get('location')
        
        # 根据时间调整
        if time_of_day:
            for rec in recommendations:
                # 示例：早晨提升新闻类内容
                if time_of_day == 'morning' and 'news' in rec.reason.lower():
                    rec.score *= 1.2
        
        # 重新排序
        recommendations.sort(key=lambda x: x.score, reverse=True)
        return recommendations
    
    def recommend_diverse(self, user_id: str, top_k: int = 10,
                          diversity_weight: float = 0.3) -> List[Recommendation]:
        """
        多样化推荐（MMR - Maximal Marginal Relevance）
        
        在相关性和多样性之间权衡
        """
        # 获取候选
        candidates = self.recommend(user_id, top_k=top_k*3)
        
        selected = []
        remaining = candidates.copy()
        
        while len(selected) < top_k and remaining:
            if not selected:
                # 选择最相关的
                best = max(remaining, key=lambda x: x.score)
            else:
                # MMR评分
                def mmr_score(rec):
                    relevance = rec.score
                    max_sim = max(
                        self._item_similarity(rec.item_id, s.item_id)
                        for s in selected
                    ) if selected else 0
                    return (1 - diversity_weight) * relevance - diversity_weight * max_sim
                
                best = max(remaining, key=mmr_score)
            
            selected.append(best)
            remaining.remove(best)
        
        return selected
    
    def _item_similarity(self, item1: str, item2: str) -> float:
        """计算两个物品的相似度"""
        # 使用内容特征相似度
        if (item1 in self.content.item_features_matrix and 
            item2 in self.content.item_features_matrix):
            return self.content._calculate_similarity(
                self.content.item_features_matrix[item1],
                self.content.item_features_matrix[item2]
            )
        return 0.0


# ============== 4. A/B测试框架 ==============

class ABTestFramework:
    """
    A/B测试框架
    管理实验、分配流量、收集指标
    """
    
    def __init__(self):
        self.experiments: Dict[str, 'Experiment'] = {}
        self.user_assignments: Dict[str, str] = {}  # user_id -> variant_id
    
    def create_experiment(self, experiment_id: str, name: str,
                         variants: List[ABTestVariant],
                         metrics: List[str]) -> 'Experiment':
        """创建A/B测试实验"""
        exp = Experiment(
            id=experiment_id,
            name=name,
            variants=variants,
            metrics=metrics,
            created_at=datetime.now()
        )
        self.experiments[experiment_id] = exp
        return exp
    
    def assign_variant(self, experiment_id: str, user_id: str) -> ABTestVariant:
        """为用户分配实验变体"""
        if experiment_id not in self.experiments:
            raise ValueError(f"实验 {experiment_id} 不存在")
        
        exp = self.experiments[experiment_id]
        
        # 检查是否已分配
        assignment_key = f"{experiment_id}:{user_id}"
        if assignment_key in self.user_assignments:
            variant_id = self.user_assignments[assignment_key]
            return next(v for v in exp.variants if v.id == variant_id)
        
        # 新分配 - 基于流量百分比
        rand = random.random()
        cumulative = 0
        
        for variant in exp.variants:
            cumulative += variant.traffic_percentage
            if rand <= cumulative:
                self.user_assignments[assignment_key] = variant.id
                exp.user_counts[variant.id] += 1
                return variant
        
        # 默认返回最后一个
        default_variant = exp.variants[-1]
        self.user_assignments[assignment_key] = default_variant.id
        exp.user_counts[default_variant.id] += 1
        return default_variant
    
    def track_event(self, experiment_id: str, user_id: str,
                   metric: str, value: float = 1.0):
        """记录用户行为事件"""
        if experiment_id not in self.experiments:
            return
        
        exp = self.experiments[experiment_id]
        assignment_key = f"{experiment_id}:{user_id}"
        variant_id = self.user_assignments.get(assignment_key)
        
        if variant_id:
            exp.record_metric(variant_id, metric, value)
    
    def get_results(self, experiment_id: str) -> Dict:
        """获取实验结果"""
        if experiment_id not in self.experiments:
            return {}
        
        exp = self.experiments[experiment_id]
        return exp.get_results()


@dataclass
class Experiment:
    """A/B测试实验"""
    id: str
    name: str
    variants: List[ABTestVariant]
    metrics: List[str]
    created_at: datetime
    user_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    metric_data: Dict[str, Dict[str, List[float]]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(list))
    )
    
    def record_metric(self, variant_id: str, metric: str, value: float):
        """记录指标"""
        self.metric_data[variant_id][metric].append(value)
    
    def get_results(self) -> Dict:
        """获取实验结果统计"""
        results = {
            'experiment_id': self.id,
            'name': self.name,
            'total_users': sum(self.user_counts.values()),
            'variant_stats': {}
        }
        
        for variant in self.variants:
            variant_stats = {
                'user_count': self.user_counts[variant.id],
                'metrics': {}
            }
            
            for metric in self.metrics:
                values = self.metric_data[variant.id].get(metric, [])
                if values:
                    variant_stats['metrics'][metric] = {
                        'count': len(values),
                        'mean': sum(values) / len(values),
                        'sum': sum(values),
                        'min': min(values),
                        'max': max(values)
                    }
            
            results['variant_stats'][variant.id] = variant_stats
        
        return results


# ============== 推荐服务 ==============

class RecommendationService:
    """推荐服务入口"""
    
    def __init__(self):
        self.recommender = HybridRecommender()
        self.ab_test = ABTestFramework()
        self.users: Dict[str, User] = {}
        self.items: Dict[str, Item] = {}
    
    def initialize(self, users: List[User], items: List[Item],
                   interactions: List[Interaction]):
        """初始化服务"""
        self.users = {u.id: u for u in users}
        self.items = {i.id: i for i in items}
        self.recommender.fit(users, items, interactions)
    
    def recommend(self, user_id: str, top_k: int = 10,
                 use_ab_test: bool = False,
                 experiment_id: Optional[str] = None) -> List[Recommendation]:
        """
        获取推荐
        
        Args:
            use_ab_test: 是否使用A/B测试
            experiment_id: A/B测试实验ID
        """
        if use_ab_test and experiment_id:
            # 分配实验变体
            variant = self.ab_test.assign_variant(experiment_id, user_id)
            
            # 根据变体调整权重
            if variant.config.get('weights'):
                self.recommender.weights = variant.config['weights']
        
        recommendations = self.recommender.recommend(user_id, top_k)
        
        # 记录曝光
        if use_ab_test and experiment_id:
            for rec in recommendations:
                self.ab_test.track_event(
                    experiment_id, user_id, 'impression', 1.0
                )
        
        return recommendations
    
    def record_feedback(self, user_id: str, item_id: str,
                       rating: float,
                       experiment_id: Optional[str] = None):
        """记录用户反馈"""
        if experiment_id:
            self.ab_test.track_event(
                experiment_id, user_id, 'rating', rating
            )
            self.ab_test.track_event(
                experiment_id, user_id, 'click', 1.0
            )


# ============== 使用示例 ==============

def example_usage():
    """使用示例"""
    # 创建示例数据
    users = [
        User(id=f"user_{i}", features={'age': 25 + i, 'gender': 'M' if i % 2 == 0 else 'F'})
        for i in range(10)
    ]
    
    items = [
        Item(
            id=f"item_{i}",
            title=f"物品{i}",
            category=random.choice(['科技', '娱乐', '体育', '新闻']),
            tags=random.sample(['热门', '新上架', '推荐', '精选'], k=2),
            features={'popularity': random.random()}
        )
        for i in range(50)
    ]
    
    # 生成交互数据
    interactions = []
    for user in users:
        for item in random.sample(items, k=random.randint(5, 15)):
            interactions.append(Interaction(
                user_id=user.id,
                item_id=item.id,
                rating=random.uniform(1, 5)
            ))
    
    # 初始化服务
    service = RecommendationService()
    service.initialize(users, items, interactions)
    
    # 示例1: 基本推荐
    print("=== 协同过滤推荐示例 ===")
    cf_recs = service.recommender.cf.user_based_recommend("user_0", top_k=5)
    for rec in cf_recs:
        item = service.items.get(rec.item_id)
        print(f"  {item.title if item else rec.item_id}: 分数={rec.score:.2f}, 原因={rec.reason}")
    
    # 示例2: 内容推荐
    print("\n=== 内容推荐示例 ===")
    content_recs = service.recommender.content.recommend("user_0", top_k=5)
    for rec in content_recs:
        item = service.items.get(rec.item_id)
        print(f"  {item.title if item else rec.item_id}: 分数={rec.score:.2f}, 原因={rec.reason}")
    
    # 示例3: 混合推荐
    print("\n=== 混合推荐示例 ===")
    hybrid_recs = service.recommend("user_0", top_k=5)
    for rec in hybrid_recs:
        item = service.items.get(rec.item_id)
        print(f"  {item.title if item else rec.item_id}: 分数={rec.score:.2f}, 方法={rec.method}")
    
    # 示例4: A/B测试
    print("\n=== A/B测试示例 ===")
    
    # 创建实验
    variants = [
        ABTestVariant(
            id="control",
            name="对照组",
            algorithm="hybrid",
            config={'weights': {'user_cf': 0.3, 'item_cf': 0.3, 'content': 0.4}},
            traffic_percentage=0.5
        ),
        ABTestVariant(
            id="treatment",
            name="实验组",
            algorithm="hybrid",
            config={'weights': {'user_cf': 0.5, 'item_cf': 0.2, 'content': 0.3}},
            traffic_percentage=0.5
        )
    ]
    
    service.ab_test.create_experiment(
        "exp_001",
        "推荐算法权重测试",
        variants,
        metrics=['impression', 'click', 'rating']
    )
    
    # 模拟用户访问
    for i in range(10):
        recs = service.recommend(
            f"user_{i}",
            use_ab_test=True,
            experiment_id="exp_001"
        )
        
        # 模拟点击
        if random.random() > 0.5 and recs:
            service.record_feedback(
                f"user_{i}",
                recs[0].item_id,
                random.uniform(3, 5),
                experiment_id="exp_001"
            )
    
    # 查看结果
    results = service.ab_test.get_results("exp_001")
    print(f"实验结果: {json.dumps(results, indent=2, default=str)}")


if __name__ == "__main__":
    example_usage()
