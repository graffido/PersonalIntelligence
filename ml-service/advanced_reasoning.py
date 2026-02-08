"""
增强推理规则模块
实现因果推理、反事实推理、时序模式挖掘、社交网络分析
"""

import json
import sqlite3
from collections import defaultdict, Counter
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Set, Callable, Any
import math
import logging

logger = logging.getLogger(__name__)

# ============== 基础数据模型 ==============

@dataclass
class CausalRelation:
    """因果关系模型"""
    cause: str
    effect: str
    confidence: float  # 置信度 0-1
    support: float     # 支持度
    temporal_gap: Optional[timedelta] = None  # 时间间隔
    evidence_count: int = 0
    metadata: Dict[str, Any] = None

@dataclass
class CounterfactualResult:
    """反事实推理结果"""
    original_outcome: str
    counterfactual_condition: str
    predicted_outcome: str
    confidence: float
    explanation: str

@dataclass
class TemporalPattern:
    """时序模式"""
    pattern: Tuple[str, ...]  # 事件序列
    support: float
    confidence: float
    frequency: int
    avg_time_gap: Optional[timedelta] = None

@dataclass
class SocialNode:
    """社交网络节点"""
    id: str
    name: str
    pagerank: float = 0.0
    centrality: float = 0.0
    connections: List[str] = None
    
    def __post_init__(self):
        if self.connections is None:
            self.connections = []


# ============== 1. 因果推理 ==============

class CausalReasoning:
    """
    因果推理引擎
    基于Granger因果关系、时间序列分析和统计相关性
    """
    
    def __init__(self, min_confidence: float = 0.6, min_support: float = 0.1):
        self.min_confidence = min_confidence
        self.min_support = min_support
        self.causal_graph: Dict[str, List[CausalRelation]] = defaultdict(list)
    
    def analyze_event_pairs(self, events: List[Dict]) -> List[CausalRelation]:
        """
        分析事件对之间的潜在因果关系
        
        Args:
            events: 事件列表，每个事件包含type和timestamp
        """
        causal_relations = []
        
        # 按事件类型分组
        event_types = defaultdict(list)
        for event in events:
            event_type = event.get('type')
            timestamp = event.get('timestamp')
            if event_type and timestamp:
                event_types[event_type].append(timestamp)
        
        # 分析每对事件类型的因果关系
        type_list = list(event_types.keys())
        for i, cause_type in enumerate(type_list):
            for effect_type in type_list[i+1:]:
                cause_times = event_types[cause_type]
                effect_times = event_types[effect_type]
                
                if len(cause_times) < 3 or len(effect_times) < 3:
                    continue
                
                # 计算时间关联性
                relation = self._calculate_temporal_correlation(
                    cause_type, effect_type,
                    cause_times, effect_times
                )
                
                if relation and relation.confidence >= self.min_confidence:
                    causal_relations.append(relation)
                    self.causal_graph[cause_type].append(relation)
        
        return causal_relations
    
    def _calculate_temporal_correlation(self, cause_type: str, effect_type: str,
                                        cause_times: List[datetime],
                                        effect_times: List[datetime]) -> Optional[CausalRelation]:
        """计算两个事件类型之间的时间关联性"""
        
        # 统计A发生后B发生的次数
        co_occurrence = 0
        time_gaps = []
        
        for cause_time in cause_times:
            # 查找在合理时间窗口内发生的effect事件
            for effect_time in effect_times:
                time_gap = effect_time - cause_time
                if timedelta(minutes=0) < time_gap < timedelta(hours=24):
                    co_occurrence += 1
                    time_gaps.append(time_gap)
                    break
        
        if not time_gaps:
            return None
        
        # 计算置信度和支持度
        confidence = co_occurrence / len(cause_times)
        support = co_occurrence / len(effect_times) if effect_times else 0
        
        # 计算平均时间间隔
        avg_gap = sum(time_gaps, timedelta()) / len(time_gaps)
        
        if confidence >= self.min_confidence and support >= self.min_support:
            return CausalRelation(
                cause=cause_type,
                effect=effect_type,
                confidence=confidence,
                support=support,
                temporal_gap=avg_gap,
                evidence_count=co_occurrence
            )
        
        return None
    
    def find_causal_chain(self, start_event: str, max_depth: int = 3) -> List[List[CausalRelation]]:
        """
        查找从起始事件开始的因果链
        
        Args:
            start_event: 起始事件类型
            max_depth: 最大链长度
        """
        chains = []
        
        def dfs(current: str, chain: List[CausalRelation], depth: int):
            if depth >= max_depth:
                return
            
            for relation in self.causal_graph.get(current, []):
                new_chain = chain + [relation]
                chains.append(new_chain)
                dfs(relation.effect, new_chain, depth + 1)
        
        dfs(start_event, [], 0)
        
        # 按总置信度排序
        chains.sort(key=lambda c: math.prod(r.confidence for r in c), reverse=True)
        return chains
    
    def explain_causation(self, cause: str, effect: str) -> Optional[str]:
        """解释因果关系"""
        for relation in self.causal_graph.get(cause, []):
            if relation.effect == effect:
                gap_str = ""
                if relation.temporal_gap:
                    gap_str = f"通常在 {relation.temporal_gap} 后发生"
                return (
                    f"'{cause}' 可能导致 '{effect}'\n"
                    f"置信度: {relation.confidence:.2%}\n"
                    f"支持度: {relation.support:.2%}\n"
                    f"证据数: {relation.evidence_count}\n"
                    f"{gap_str}"
                )
        return None


# ============== 2. 反事实推理 ==============

class CounterfactualReasoning:
    """
    反事实推理引擎
    "如果...会怎样"类型的推理
    """
    
    def __init__(self, causal_engine: CausalReasoning):
        self.causal_engine = causal_engine
        self.historical_outcomes: Dict[str, List[Dict]] = defaultdict(list)
    
    def add_historical_case(self, conditions: Dict, outcome: str):
        """添加历史案例用于反事实推理"""
        case_key = self._hash_conditions(conditions)
        self.historical_outcomes[outcome].append({
            'conditions': conditions,
            'timestamp': datetime.now()
        })
    
    def _hash_conditions(self, conditions: Dict) -> str:
        """将条件字典转换为哈希键"""
        sorted_items = sorted(conditions.items())
        return json.dumps(sorted_items, sort_keys=True)
    
    def infer(self, original_conditions: Dict, 
              counterfactual_condition: str,
              counterfactual_value: Any) -> CounterfactualResult:
        """
        执行反事实推理
        
        Args:
            original_conditions: 原始条件
            counterfactual_condition: 要改变的条件名称
            counterfactual_value: 改变后的值
        """
        # 构建反事实条件
        cf_conditions = original_conditions.copy()
        cf_conditions[counterfactual_condition] = counterfactual_value
        
        # 基于历史数据预测结果
        original_outcome = self._predict_outcome(original_conditions)
        cf_outcome = self._predict_outcome(cf_conditions)
        
        # 计算置信度
        confidence = self._calculate_confidence(cf_conditions, cf_outcome)
        
        # 生成解释
        explanation = self._generate_explanation(
            original_conditions, original_outcome,
            counterfactual_condition, counterfactual_value,
            cf_outcome
        )
        
        return CounterfactualResult(
            original_outcome=original_outcome,
            counterfactual_condition=f"{counterfactual_condition} = {counterfactual_value}",
            predicted_outcome=cf_outcome,
            confidence=confidence,
            explanation=explanation
        )
    
    def _predict_outcome(self, conditions: Dict) -> str:
        """基于历史数据预测结果"""
        # 简单实现：找到最相似的历史案例
        best_match = None
        best_score = 0
        
        for outcome, cases in self.historical_outcomes.items():
            for case in cases[-100:]:  # 只考虑最近的100个案例
                score = self._similarity_score(conditions, case['conditions'])
                if score > best_score:
                    best_score = score
                    best_match = outcome
        
        return best_match or "未知"
    
    def _similarity_score(self, cond1: Dict, cond2: Dict) -> float:
        """计算条件相似度"""
        if not cond1 or not cond2:
            return 0.0
        
        common_keys = set(cond1.keys()) & set(cond2.keys())
        if not common_keys:
            return 0.0
        
        matches = sum(1 for k in common_keys if cond1[k] == cond2[k])
        return matches / len(common_keys)
    
    def _calculate_confidence(self, conditions: Dict, outcome: str) -> float:
        """计算预测置信度"""
        cases = self.historical_outcomes.get(outcome, [])
        if not cases:
            return 0.0
        
        # 基于相似案例数量计算置信度
        similar_cases = sum(
            1 for c in cases 
            if self._similarity_score(conditions, c['conditions']) > 0.5
        )
        
        return min(similar_cases / 10, 1.0)  # 最多10个相似案例达到100%置信度
    
    def _generate_explanation(self, original_conditions: Dict, original_outcome: str,
                              cf_condition: str, cf_value: Any, cf_outcome: str) -> str:
        """生成反事实推理解释"""
        if original_outcome == cf_outcome:
            return f"改变 {cf_condition} 似乎不会影响结果，仍会是 {original_outcome}"
        
        return (
            f"原始条件下结果是 {original_outcome}。\n"
            f"如果将 {cf_condition} 改为 {cf_value}，"
            f"结果可能会变成 {cf_outcome}。\n"
            f"这是基于历史相似案例的模式推断得出的。"
        )


# ============== 3. 时序模式挖掘 ==============

class TemporalPatternMining:
    """
    时序模式挖掘
    实现Apriori和FP-Growth算法用于发现频繁序列模式
    """
    
    def __init__(self, min_support: int = 2, max_gap: timedelta = timedelta(hours=1)):
        self.min_support = min_support
        self.max_gap = max_gap
    
    def mine_patterns(self, event_sequences: List[List[str]]) -> List[TemporalPattern]:
        """
        从事件序列中挖掘频繁模式
        
        Args:
            event_sequences: 事件类型序列列表
        """
        # 使用Apriori算法的简化版本
        patterns = self._apriori_mining(event_sequences)
        return patterns
    
    def _apriori_mining(self, sequences: List[List[str]]) -> List[TemporalPattern]:
        """Apriori算法实现"""
        patterns = []
        
        # 1. 找出所有1-项集
        item_counts = Counter()
        for seq in sequences:
            unique_items = set(seq)
            for item in unique_items:
                item_counts[item] += 1
        
        # 2. 生成频繁1-项集
        frequent_1 = [
            (item, count) for item, count in item_counts.items()
            if count >= self.min_support
        ]
        
        # 3. 迭代生成更长的模式
        current_patterns = [([item], count) for item, count in frequent_1]
        
        while current_patterns:
            patterns.extend(current_patterns)
            current_patterns = self._generate_candidates(
                current_patterns, sequences
            )
        
        # 转换为TemporalPattern对象
        result = []
        for pattern_list, count in patterns:
            if len(pattern_list) >= 2:  # 只返回长度>=2的模式
                confidence = self._calculate_confidence(pattern_list, sequences)
                result.append(TemporalPattern(
                    pattern=tuple(pattern_list),
                    support=count / len(sequences),
                    confidence=confidence,
                    frequency=count
                ))
        
        return result
    
    def _generate_candidates(self, current_patterns: List[Tuple[List[str], int]],
                             sequences: List[List[str]]) -> List[Tuple[List[str], int]]:
        """生成候选模式"""
        candidates = []
        pattern_items = [p[0] for p in current_patterns]
        
        # 合并模式
        for i, pattern1 in enumerate(pattern_items):
            for pattern2 in pattern_items[i+1:]:
                # 尝试合并两个模式
                if pattern1[:-1] == pattern2[:-1]:
                    new_pattern = pattern1 + [pattern2[-1]]
                    count = self._count_pattern(new_pattern, sequences)
                    if count >= self.min_support:
                        candidates.append((new_pattern, count))
        
        return candidates
    
    def _count_pattern(self, pattern: List[str], sequences: List[List[str]]) -> int:
        """计算模式在序列中的出现次数"""
        count = 0
        for seq in sequences:
            if self._contains_pattern(pattern, seq):
                count += 1
        return count
    
    def _contains_pattern(self, pattern: List[str], sequence: List[str]) -> bool:
        """检查序列是否包含模式"""
        if not pattern:
            return True
        if len(pattern) > len(sequence):
            return False
        
        # 使用简单的子序列匹配
        pattern_idx = 0
        for item in sequence:
            if item == pattern[pattern_idx]:
                pattern_idx += 1
                if pattern_idx == len(pattern):
                    return True
        return False
    
    def _calculate_confidence(self, pattern: List[str], sequences: List[List[str]]) -> float:
        """计算模式的置信度"""
        if len(pattern) < 2:
            return 1.0
        
        # P(B|A) = count(A,B) / count(A)
        pattern_count = self._count_pattern(pattern, sequences)
        prefix_count = self._count_pattern(pattern[:-1], sequences)
        
        return pattern_count / prefix_count if prefix_count > 0 else 0.0
    
    def fp_growth_mining(self, sequences: List[List[str]]) -> List[TemporalPattern]:
        """
        FP-Growth算法实现（简化版）
        对于大规模数据更高效
        """
        # 构建FP-Tree
        fp_tree = self._build_fp_tree(sequences)
        
        # 递归挖掘
        patterns = self._fp_growth(fp_tree, [])
        
        return patterns
    
    def _build_fp_tree(self, sequences: List[List[str]]) -> Dict:
        """构建FP-Tree"""
        # 简化实现：使用前缀树结构
        tree = {'count': 0, 'children': {}}
        
        for seq in sequences:
            current = tree
            for item in seq:
                if item not in current['children']:
                    current['children'][item] = {'count': 0, 'children': {}}
                current = current['children'][item]
                current['count'] += 1
        
        return tree
    
    def _fp_growth(self, tree: Dict, prefix: List[str]) -> List[TemporalPattern]:
        """递归FP-Growth挖掘"""
        patterns = []
        
        for item, node in tree.get('children', {}).items():
            if node['count'] >= self.min_support:
                new_pattern = prefix + [item]
                confidence = self._calculate_tree_confidence(tree, new_pattern)
                patterns.append(TemporalPattern(
                    pattern=tuple(new_pattern),
                    support=node['count'],
                    confidence=confidence,
                    frequency=node['count']
                ))
                
                # 递归
                child_patterns = self._fp_growth(node, new_pattern)
                patterns.extend(child_patterns)
        
        return patterns
    
    def _calculate_tree_confidence(self, tree: Dict, pattern: List[str]) -> float:
        """从FP-Tree计算置信度"""
        if len(pattern) < 2:
            return 1.0
        
        # 查找模式计数
        pattern_count = self._find_pattern_count(tree, pattern)
        prefix_count = self._find_pattern_count(tree, pattern[:-1])
        
        return pattern_count / prefix_count if prefix_count > 0 else 0.0
    
    def _find_pattern_count(self, tree: Dict, pattern: List[str]) -> int:
        """在FP-Tree中查找模式计数"""
        current = tree
        for item in pattern:
            if item not in current.get('children', {}):
                return 0
            current = current['children'][item]
        return current.get('count', 0)


# ============== 4. 社交网络分析 ==============

class SocialNetworkAnalysis:
    """
    社交网络分析
    PageRank、中心性计算、社区发现
    """
    
    def __init__(self, damping_factor: float = 0.85, max_iterations: int = 100):
        self.damping_factor = damping_factor
        self.max_iterations = max_iterations
        self.graph: Dict[str, Set[str]] = defaultdict(set)
        self.nodes: Dict[str, SocialNode] = {}
    
    def add_connection(self, person_a: str, person_b: str, 
                       name_a: str = None, name_b: str = None):
        """添加社交连接"""
        self.graph[person_a].add(person_b)
        self.graph[person_b].add(person_a)  # 无向图
        
        # 确保节点存在
        if person_a not in self.nodes:
            self.nodes[person_a] = SocialNode(
                id=person_a,
                name=name_a or person_a
            )
        if person_b not in self.nodes:
            self.nodes[person_b] = SocialNode(
                id=person_b,
                name=name_b or person_b
            )
    
    def add_interactions(self, interactions: List[Tuple[str, str, int]]):
        """
        批量添加交互数据
        
        Args:
            interactions: [(person_a, person_b, weight), ...]
        """
        for person_a, person_b, weight in interactions:
            for _ in range(weight):
                self.add_connection(person_a, person_b)
    
    def calculate_pagerank(self) -> Dict[str, float]:
        """
        计算PageRank值
        用于衡量节点在社交网络中的重要性
        """
        if not self.nodes:
            return {}
        
        n = len(self.nodes)
        pagerank = {node_id: 1.0 / n for node_id in self.nodes}
        
        for iteration in range(self.max_iterations):
            new_pagerank = {}
            
            for node_id in self.nodes:
                # 随机跳转部分
                rank = (1 - self.damping_factor) / n
                
                # 从邻居节点获得的贡献
                for neighbor in self.graph[node_id]:
                    if neighbor in pagerank and len(self.graph[neighbor]) > 0:
                        rank += self.damping_factor * pagerank[neighbor] / len(self.graph[neighbor])
                
                new_pagerank[node_id] = rank
            
            # 归一化
            total = sum(new_pagerank.values())
            if total > 0:
                new_pagerank = {k: v / total for k, v in new_pagerank.items()}
            
            # 检查收敛
            diff = sum(abs(new_pagerank[k] - pagerank[k]) for k in pagerank)
            pagerank = new_pagerank
            
            if diff < 1e-6:
                logger.info(f"PageRank在 {iteration + 1} 次迭代后收敛")
                break
        
        # 更新节点PageRank
        for node_id, rank in pagerank.items():
            self.nodes[node_id].pagerank = rank
        
        return pagerank
    
    def calculate_centrality(self) -> Dict[str, float]:
        """
        计算节点的中心性（度中心性）
        """
        if not self.nodes:
            return {}
        
        max_degree = max(len(neighbors) for neighbors in self.graph.values()) if self.graph else 1
        
        centrality = {}
        for node_id in self.nodes:
            degree = len(self.graph[node_id])
            centrality[node_id] = degree / max_degree if max_degree > 0 else 0
            self.nodes[node_id].centrality = centrality[node_id]
        
        return centrality
    
    def find_communities(self) -> List[Set[str]]:
        """
        使用简单的标签传播算法发现社区
        """
        if not self.nodes:
            return []
        
        # 初始化：每个节点一个社区
        labels = {node_id: i for i, node_id in enumerate(self.nodes)}
        
        for iteration in range(self.max_iterations):
            changed = False
            
            for node_id in self.nodes:
                if not self.graph[node_id]:
                    continue
                
                # 统计邻居的标签
                neighbor_labels = Counter(
                    labels[neighbor] 
                    for neighbor in self.graph[node_id]
                )
                
                # 选择最常见的标签
                if neighbor_labels:
                    new_label = neighbor_labels.most_common(1)[0][0]
                    if labels[node_id] != new_label:
                        labels[node_id] = new_label
                        changed = True
            
            if not changed:
                break
        
        # 收集社区
        communities: Dict[int, Set[str]] = defaultdict(set)
        for node_id, label in labels.items():
            communities[label].add(node_id)
        
        return list(communities.values())
    
    def find_influencers(self, top_n: int = 10) -> List[SocialNode]:
        """找出影响力最大的节点"""
        # 确保已计算PageRank
        self.calculate_pagerank()
        self.calculate_centrality()
        
        # 综合评分
        for node in self.nodes.values():
            node.connections = list(self.graph.get(node.id, []))
        
        sorted_nodes = sorted(
            self.nodes.values(),
            key=lambda n: (n.pagerank + n.centrality) / 2,
            reverse=True
        )
        
        return sorted_nodes[:top_n]
    
    def find_bridges(self) -> List[str]:
        """
        查找网络中的桥接节点（连接不同社区的节点）
        """
        bridges = []
        communities = self.find_communities()
        
        # 构建社区映射
        node_to_community = {}
        for i, community in enumerate(communities):
            for node_id in community:
                node_to_community[node_id] = i
        
        # 查找连接多个社区的节点
        for node_id in self.nodes:
            neighbor_communities = set()
            for neighbor in self.graph[node_id]:
                if neighbor in node_to_community:
                    neighbor_communities.add(node_to_community[neighbor])
            
            if len(neighbor_communities) > 1:
                bridges.append(node_id)
        
        return bridges
    
    def get_network_stats(self) -> Dict:
        """获取网络统计信息"""
        if not self.nodes:
            return {}
        
        pagerank = self.calculate_pagerank()
        centrality = self.calculate_centrality()
        communities = self.find_communities()
        
        return {
            "node_count": len(self.nodes),
            "edge_count": sum(len(neighbors) for neighbors in self.graph.values()) // 2,
            "avg_degree": sum(len(neighbors) for neighbors in self.graph.values()) / len(self.nodes),
            "community_count": len(communities),
            "avg_community_size": len(self.nodes) / len(communities) if communities else 0,
            "max_pagerank": max(pagerank.values()) if pagerank else 0,
            "min_pagerank": min(pagerank.values()) if pagerank else 0
        }


# ============== 综合推理引擎 ==============

class AdvancedReasoningEngine:
    """综合推理引擎，整合所有推理能力"""
    
    def __init__(self, db_path: str = "reasoning_data.db"):
        self.causal = CausalReasoning()
        self.counterfactual = CounterfactualReasoning(self.causal)
        self.temporal = TemporalPatternMining()
        self.social = SocialNetworkAnalysis()
        self.db_path = db_path
    
    def analyze_events(self, events: List[Dict]) -> Dict:
        """
        综合分析事件数据
        
        Returns:
            包含因果关系、时序模式的分析结果
        """
        result = {
            "causal_relations": [],
            "temporal_patterns": [],
            "insights": []
        }
        
        # 因果分析
        causal_relations = self.causal.analyze_event_pairs(events)
        result["causal_relations"] = [asdict(r) for r in causal_relations]
        
        # 时序模式挖掘
        event_sequences = self._group_into_sequences(events)
        patterns = self.temporal.mine_patterns(event_sequences)
        result["temporal_patterns"] = [asdict(p) for p in patterns[:10]]  # 前10个模式
        
        # 生成洞察
        insights = self._generate_insights(causal_relations, patterns)
        result["insights"] = insights
        
        return result
    
    def _group_into_sequences(self, events: List[Dict]) -> List[List[str]]:
        """将事件分组为序列"""
        # 按时间窗口分组
        sorted_events = sorted(events, key=lambda e: e.get('timestamp', datetime.min))
        
        sequences = []
        current_sequence = []
        last_time = None
        
        for event in sorted_events:
            current_time = event.get('timestamp')
            
            if last_time and current_time:
                gap = current_time - last_time
                if gap > timedelta(hours=2):  # 超过2小时视为新序列
                    if current_sequence:
                        sequences.append(current_sequence)
                    current_sequence = []
            
            current_sequence.append(event.get('type', 'unknown'))
            last_time = current_time
        
        if current_sequence:
            sequences.append(current_sequence)
        
        return sequences
    
    def _generate_insights(self, causal_relations: List[CausalRelation],
                          patterns: List[TemporalPattern]) -> List[str]:
        """生成分析洞察"""
        insights = []
        
        # 强因果关系
        strong_causal = [r for r in causal_relations if r.confidence > 0.8]
        for r in strong_causal[:3]:
            insights.append(
                f"发现强因果关系: {r.cause} → {r.effect} (置信度: {r.confidence:.1%})"
            )
        
        # 频繁模式
        frequent = [p for p in patterns if p.frequency >= 5]
        for p in frequent[:3]:
            pattern_str = " → ".join(p.pattern)
            insights.append(
                f"频繁模式: {pattern_str} (出现{p.frequency}次)"
            )
        
        return insights
    
    def analyze_social_network(self, interactions: List[Tuple[str, str, int]]) -> Dict:
        """分析社交网络"""
        self.social.add_interactions(interactions)
        
        pagerank = self.social.calculate_pagerank()
        centrality = self.social.calculate_centrality()
        communities = self.social.find_communities()
        influencers = self.social.find_influencers(5)
        bridges = self.social.find_bridges()
        
        return {
            "stats": self.social.get_network_stats(),
            "top_influencers": [
                {"id": n.id, "name": n.name, "pagerank": n.pagerank}
                for n in influencers
            ],
            "communities": [
                {"id": i, "members": list(members)}
                for i, members in enumerate(communities[:5])  # 前5个社区
            ],
            "bridges": bridges[:10],
            "centrality": dict(list(centrality.items())[:20])
        }
    
    def counterfactual_query(self, original_conditions: Dict,
                            counterfactual_condition: str,
                            counterfactual_value: Any) -> Dict:
        """反事实查询接口"""
        result = self.counterfactual.infer(
            original_conditions, counterfactual_condition, counterfactual_value
        )
        return asdict(result)


# ============== 使用示例 ==============

def example_usage():
    """使用示例"""
    engine = AdvancedReasoningEngine()
    
    # 示例1: 因果推理
    print("=== 因果推理示例 ===")
    events = [
        {'type': '喝咖啡', 'timestamp': datetime.now() - timedelta(hours=5)},
        {'type': '失眠', 'timestamp': datetime.now() - timedelta(hours=2)},
        {'type': '喝咖啡', 'timestamp': datetime.now() - timedelta(days=1, hours=5)},
        {'type': '失眠', 'timestamp': datetime.now() - timedelta(days=1, hours=2)},
        {'type': '运动', 'timestamp': datetime.now() - timedelta(days=2, hours=6)},
        {'type': '失眠', 'timestamp': datetime.now() - timedelta(days=2, hours=1)},
    ]
    
    causal_relations = engine.causal.analyze_event_pairs(events)
    for r in causal_relations:
        print(f"{r.cause} → {r.effect}: 置信度={r.confidence:.2f}, 支持度={r.support:.2f}")
    
    # 示例2: 时序模式挖掘
    print("\n=== 时序模式挖掘示例 ===")
    sequences = [
        ['早餐', '工作', '午餐', '工作', '晚餐'],
        ['早餐', '运动', '午餐', '工作', '晚餐'],
        ['早餐', '工作', '午餐', '会议', '晚餐'],
        ['早餐', '工作', '午餐', '工作', '运动'],
        ['早餐', '阅读', '午餐', '工作', '晚餐'],
    ]
    
    patterns = engine.temporal.mine_patterns(sequences)
    for p in patterns[:5]:
        pattern_str = " → ".join(p.pattern)
        print(f"模式: {pattern_str}, 支持度={p.support:.2f}, 置信度={p.confidence:.2f}")
    
    # 示例3: 社交网络分析
    print("\n=== 社交网络分析示例 ===")
    interactions = [
        ('Alice', 'Bob', 5),
        ('Alice', 'Charlie', 3),
        ('Bob', 'David', 4),
        ('Charlie', 'David', 2),
        ('David', 'Eve', 6),
        ('Eve', 'Frank', 3),
        ('Frank', 'Alice', 2),
    ]
    
    result = engine.analyze_social_network(interactions)
    print(f"网络统计: {result['stats']}")
    print(f"\n重要人物:")
    for inf in result['top_influencers']:
        print(f"  {inf['name']}: PageRank={inf['pagerank']:.4f}")
    
    # 示例4: 综合分析
    print("\n=== 综合分析示例 ===")
    analysis = engine.analyze_events(events)
    print(f"发现 {len(analysis['causal_relations'])} 个因果关系")
    print(f"发现 {len(analysis['temporal_patterns'])} 个时序模式")
    print("洞察:")
    for insight in analysis['insights']:
        print(f"  - {insight}")


if __name__ == "__main__":
    example_usage()
