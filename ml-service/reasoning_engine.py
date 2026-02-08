#!/usr/bin/env python3
"""
POS 本地推理引擎 (Python版本)
基于本体的轻量级推理，减少LLM调用
"""

from typing import List, Dict, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
from datetime import datetime, timedelta
import re

class RuleType(Enum):
    TRANSITIVE = "transitive"      # 传递性推理
    SYMMETRIC = "symmetric"        # 对称性
    TEMPORAL = "temporal"          # 时间约束
    SPATIAL = "spatial"            # 空间约束
    PATTERN = "pattern"            # 模式匹配
    CONFLICT = "conflict"          # 冲突检测
    CUSTOM = "custom"              # 自定义

@dataclass
class Concept:
    """本体概念"""
    id: str
    type: str
    label: str
    properties: Dict[str, Any] = field(default_factory=dict)
    relations: List[Dict] = field(default_factory=list)

@dataclass
class Memory:
    """记忆"""
    id: str
    content: str
    timestamp: datetime
    entities: List[Dict]
    location: Optional[Dict] = None
    emotions: List[Dict] = field(default_factory=list)
    ontology_bindings: List[str] = field(default_factory=list)

@dataclass
class InferenceResult:
    """推理结果"""
    rule_id: str
    rule_name: str
    result_type: str  # "new_relation", "conflict", "suggestion", "pattern", "prediction"
    description: str
    confidence: float
    involved_concepts: List[str] = field(default_factory=list)
    involved_memories: List[str] = field(default_factory=list)
    suggested_action: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Pattern:
    """发现的模式"""
    pattern_type: str
    description: str
    confidence: float
    frequency: int
    examples: List[str]
    prediction: Optional[str] = None
    next_occurrence: Optional[datetime] = None

class LocalReasoningEngine:
    """本地推理引擎"""
    
    def __init__(self):
        self.concepts: Dict[str, Concept] = {}
        self.memories: Dict[str, Memory] = {}
        self.rules: List[Dict] = []
        self._register_builtin_rules()
    
    def _register_builtin_rules(self):
        """注册内置推理规则"""
        self.rules = [
            {
                "id": "friend_of_friend",
                "name": "朋友的朋友",
                "type": RuleType.TRANSITIVE,
                "description": "朋友的朋友可能是潜在联系人",
                "apply": self._apply_friend_of_friend
            },
            {
                "id": "schedule_conflict",
                "name": "日程冲突检测",
                "type": RuleType.CONFLICT,
                "description": "检测时间或空间上的冲突",
                "apply": self._apply_schedule_conflict
            },
            {
                "id": "location_reachable",
                "name": "地点可达性",
                "type": RuleType.SPATIAL,
                "description": "检查是否能在时间限制内到达目的地",
                "apply": self._apply_location_reachable
            },
            {
                "id": "habit_pattern",
                "name": "习惯模式识别",
                "type": RuleType.PATTERN,
                "description": "从重复行为中发现习惯模式",
                "apply": self._apply_habit_pattern
            },
            {
                "id": "time_reminder",
                "name": "时间提醒",
                "type": RuleType.TEMPORAL,
                "description": "基于历史模式提醒即将到来的事件",
                "apply": self._apply_time_reminder
            },
            {
                "id": "social_strength",
                "name": "社交关系强度",
                "type": RuleType.PATTERN,
                "description": "基于互动频率计算关系强度",
                "apply": self._apply_social_strength
            },
            {
                "id": "location_pattern",
                "name": "地点模式",
                "type": RuleType.PATTERN,
                "description": "发现常去的地点组合",
                "apply": self._apply_location_pattern
            },
        ]
    
    def add_concept(self, concept: Concept):
        """添加概念"""
        self.concepts[concept.id] = concept
    
    def add_memory(self, memory: Memory):
        """添加记忆"""
        self.memories[memory.id] = memory
    
    def infer(self, context: Optional[Dict] = None) -> List[InferenceResult]:
        """
        执行推理
        
        Args:
            context: 可选的上下文信息，如当前时间、位置等
        """
        results = []
        
        for rule in self.rules:
            try:
                rule_results = rule["apply"](context)
                results.extend(rule_results)
            except Exception as e:
                print(f"Rule {rule['id']} failed: {e}")
        
        # 按置信度排序
        results.sort(key=lambda x: x.confidence, reverse=True)
        return results
    
    def infer_for_concept(self, concept_id: str) -> List[InferenceResult]:
        """针对特定概念进行推理"""
        results = []
        
        # 传递性推理
        if concept_id in self.concepts:
            concept = self.concepts[concept_id]
            # 查找所有KNOWS关系
            knows_relations = [r for r in concept.relations if r.get("type") == "KNOWS"]
            
            for rel in knows_relations:
                friend_id = rel.get("target")
                if friend_id in self.concepts:
                    friend = self.concepts[friend_id]
                    # 查找朋友的朋友
                    for friend_rel in friend.relations:
                        if friend_rel.get("type") == "KNOWS":
                            fof_id = friend_rel.get("target")
                            if fof_id != concept_id and fof_id in self.concepts:
                                fof = self.concepts[fof_id]
                                results.append(InferenceResult(
                                    rule_id="friend_of_friend",
                                    rule_name="朋友的朋友",
                                    result_type="suggestion",
                                    description=f"{concept.label} 可能认识 {fof.label} (通过 {friend.label} 介绍)",
                                    confidence=0.7,
                                    involved_concepts=[concept_id, fof_id],
                                    suggested_action=f"建议引荐 {concept.label} 和 {fof.label} 认识"
                                ))
        
        return results
    
    def detect_conflicts(self, time_window_hours: int = 48) -> List[InferenceResult]:
        """检测冲突"""
        results = []
        now = datetime.now()
        window_end = now + timedelta(hours=time_window_hours)
        
        # 获取时间窗口内的记忆
        upcoming = [
            m for m in self.memories.values()
            if now <= m.timestamp <= window_end
        ]
        
        # 检查时间冲突
        for i, mem1 in enumerate(upcoming):
            for mem2 in upcoming[i+1:]:
                # 检查时间重叠
                time_diff = abs((mem1.timestamp - mem2.timestamp).total_seconds())
                
                if time_diff < 3600:  # 1小时内
                    # 检查是否为同一地点
                    same_location = False
                    if mem1.location and mem2.location:
                        dist = self._calculate_distance(mem1.location, mem2.location)
                        same_location = dist < 500  # 500米内
                    
                    if not same_location:
                        results.append(InferenceResult(
                            rule_id="schedule_conflict",
                            rule_name="日程冲突",
                            result_type="conflict",
                            description=f"时间冲突: {mem1.content[:30]}... 与 {mem2.content[:30]}...",
                            confidence=0.9,
                            involved_memories=[mem1.id, mem2.id],
                            suggested_action="需要重新安排其中一个日程",
                            metadata={"severity": "high", "time_diff_minutes": time_diff / 60}
                        ))
        
        return results
    
    def discover_patterns(self, min_frequency: int = 2) -> List[Pattern]:
        """发现行为模式"""
        patterns = []
        
        # 按概念分组记忆
        concept_memories: Dict[str, List[Memory]] = {}
        for mem in self.memories.values():
            for concept_id in mem.ontology_bindings:
                if concept_id not in concept_memories:
                    concept_memories[concept_id] = []
                concept_memories[concept_id].append(mem)
        
        # 查找重复模式
        for concept_id, memories in concept_memories.items():
            if len(memories) >= min_frequency:
                concept = self.concepts.get(concept_id)
                if not concept:
                    continue
                
                # 分析时间模式
                hours = [m.timestamp.hour for m in memories]
                hour_counts = {}
                for h in hours:
                    hour_counts[h] = hour_counts.get(h, 0) + 1
                
                # 找出最常见的小时
                if hour_counts:
                    common_hour = max(hour_counts, key=hour_counts.get)
                    frequency = hour_counts[common_hour]
                    
                    if frequency >= min_frequency:
                        time_desc = self._hour_to_time_of_day(common_hour)
                        
                        # 预测下次发生时间
                        last_time = max(m.timestamp for m in memories)
                        next_time = last_time + timedelta(days=7)  # 假设每周重复
                        
                        patterns.append(Pattern(
                            pattern_type="temporal",
                            description=f"经常在{time_desc}与{concept.label}相关的活动",
                            confidence=frequency / len(memories),
                            frequency=frequency,
                            examples=[m.id for m in memories],
                            prediction=f"下次可能在{time_desc}进行相关活动",
                            next_occurrence=next_time
                        ))
        
        return patterns
    
    def query_with_reasoning(self, query: str, query_entities: List[Dict]) -> Dict:
        """
        带推理的查询处理
        
        这是核心函数，处理用户查询并应用本地推理
        """
        results = {
            "direct_matches": [],
            "inferred_matches": [],
            "suggestions": [],
            "reasoning_path": []
        }
        
        # 1. 直接匹配 - 基于实体
        entity_ids = [e.get("text") for e in query_entities]
        for mem in self.memories.values():
            mem_entities = [e.get("text") for e in mem.entities]
            if any(e in mem_entities for e in entity_ids):
                results["direct_matches"].append({
                    "memory_id": mem.id,
                    "content": mem.content,
                    "timestamp": mem.timestamp.isoformat(),
                    "match_type": "direct"
                })
        
        # 2. 推理匹配 - 基于本体关系
        for entity_text in entity_ids:
            # 查找相关概念
            related_concepts = self._find_related_concepts(entity_text)
            for concept_id in related_concepts:
                # 查找绑定到该概念的记忆
                for mem in self.memories.values():
                    if concept_id in mem.ontology_bindings:
                        results["inferred_matches"].append({
                            "memory_id": mem.id,
                            "content": mem.content,
                            "timestamp": mem.timestamp.isoformat(),
                            "match_type": "inferred",
                            "reasoning": f"通过概念关联: {concept_id}"
                        })
        
        # 3. 应用推理规则生成建议
        inferences = self.infer()
        for inf in inferences[:3]:  # 取前3个
            results["suggestions"].append({
                "type": inf.result_type,
                "description": inf.description,
                "confidence": inf.confidence,
                "action": inf.suggested_action
            })
        
        return results
    
    # ============ 具体规则实现 ============
    
    def _apply_friend_of_friend(self, context: Optional[Dict]) -> List[InferenceResult]:
        """应用朋友的朋友规则"""
        results = []
        
        # 获取所有PERSON类型的概念
        persons = [c for c in self.concepts.values() if c.type == "PERSON"]
        
        for person in persons:
            # 查找KNOWS关系
            knows = [r for r in person.relations if r.get("type") == "KNOWS"]
            for rel in knows:
                friend_id = rel.get("target")
                if friend_id in self.concepts:
                    friend = self.concepts[friend_id]
                    # 查找朋友的朋友
                    for friend_rel in friend.relations:
                        if friend_rel.get("type") == "KNOWS":
                            fof_id = friend_rel.get("target")
                            if fof_id != person.id and fof_id in self.concepts:
                                fof = self.concepts[fof_id]
                                results.append(InferenceResult(
                                    rule_id="friend_of_friend",
                                    rule_name="朋友的朋友",
                                    result_type="suggestion",
                                    description=f"{person.label} 可能认识 {fof.label} (共同好友: {friend.label})",
                                    confidence=0.6,
                                    involved_concepts=[person.id, fof_id],
                                    suggested_action=f"考虑引荐 {person.label} 和 {fof.label} 认识"
                                ))
        
        return results
    
    def _apply_schedule_conflict(self, context: Optional[Dict]) -> List[InferenceResult]:
        """应用日程冲突检测"""
        return self.detect_conflicts(time_window_hours=48)
    
    def _apply_location_reachable(self, context: Optional[Dict]) -> List[InferenceResult]:
        """应用地点可达性检查"""
        results = []
        
        if not context:
            return results
        
        current_location = context.get("current_location")
        upcoming_events = [
            m for m in self.memories.values()
            if m.timestamp > datetime.now() and m.location
        ]
        
        for event in upcoming_events:
            time_until = (event.timestamp - datetime.now()).total_seconds() / 60  # 分钟
            if time_until < 60:  # 1小时内
                distance = self._calculate_distance(current_location, event.location)
                # 假设平均速度 30km/h = 500m/min
                travel_time = distance / 500
                
                if travel_time > time_until:
                    results.append(InferenceResult(
                        rule_id="location_reachable",
                        rule_name="地点可达性警告",
                        result_type="conflict",
                        description=f"可能无法按时到达: {event.content[:30]}...",
                        confidence=0.85,
                        involved_memories=[event.id],
                        suggested_action="建议提前出发或重新安排",
                        metadata={
                            "distance_meters": distance,
                            "travel_time_minutes": travel_time,
                            "time_available": time_until
                        }
                    ))
        
        return results
    
    def _apply_habit_pattern(self, context: Optional[Dict]) -> List[InferenceResult]:
        """应用习惯模式识别"""
        results = []
        patterns = self.discover_patterns(min_frequency=3)
        
        for pattern in patterns:
            if pattern.confidence > 0.6:
                results.append(InferenceResult(
                    rule_id="habit_pattern",
                    rule_name="习惯模式",
                    result_type="pattern",
                    description=pattern.description,
                    confidence=pattern.confidence,
                    involved_memories=pattern.examples,
                    suggested_action=pattern.prediction
                ))
        
        return results
    
    def _apply_time_reminder(self, context: Optional[Dict]) -> List[InferenceResult]:
        """应用时间提醒"""
        results = []
        now = datetime.now()
        
        # 检查即将到来的习惯事件
        patterns = self.discover_patterns(min_frequency=2)
        for pattern in patterns:
            if pattern.next_occurrence:
                time_until = (pattern.next_occurrence - now).total_seconds() / 3600  # 小时
                if 0 < time_until < 24:  # 24小时内
                    results.append(InferenceResult(
                        rule_id="time_reminder",
                        rule_name="时间提醒",
                        result_type="suggestion",
                        description=f"根据习惯，{pattern.description}",
                        confidence=pattern.confidence,
                        suggested_action=f"预计在 {pattern.next_occurrence.strftime('%Y-%m-%d %H:%M')} 有相关活动"
                    ))
        
        return results
    
    def _apply_social_strength(self, context: Optional[Dict]) -> List[InferenceResult]:
        """应用社交关系强度计算"""
        results = []
        
        # 计算每对人物的互动频率
        person_interactions: Dict[str, Dict[str, int]] = {}
        
        for mem in self.memories.values():
            persons = [e.get("text") for e in mem.entities if e.get("label") == "PERSON"]
            for i, p1 in enumerate(persons):
                for p2 in persons[i+1:]:
                    if p1 not in person_interactions:
                        person_interactions[p1] = {}
                    person_interactions[p1][p2] = person_interactions[p1].get(p2, 0) + 1
        
        # 找出高频互动
        for p1, interactions in person_interactions.items():
            for p2, count in interactions.items():
                if count >= 5:  # 5次以上
                    results.append(InferenceResult(
                        rule_id="social_strength",
                        rule_name="强社交关系",
                        result_type="pattern",
                        description=f"{p1} 和 {p2} 近期互动频繁 ({count}次)",
                        confidence=min(count / 10, 1.0),
                        metadata={"interaction_count": count}
                    ))
        
        return results
    
    def _apply_location_pattern(self, context: Optional[Dict]) -> List[InferenceResult]:
        """应用地点模式识别"""
        results = []
        
        # 查找常一起出现的地点
        location_pairs: Dict[str, int] = {}
        
        for mem in self.memories.values():
            if mem.location:
                loc_name = mem.location.get("name", "")
                # 检查同一天的其他记忆
                same_day = [
                    m for m in self.memories.values()
                    if m.location and m.id != mem.id
                    and abs((m.timestamp - mem.timestamp).days) == 0
                ]
                for other in same_day:
                    other_name = other.location.get("name", "")
                    pair = tuple(sorted([loc_name, other_name]))
                    location_pairs[pair] = location_pairs.get(pair, 0) + 1
        
        # 找出高频地点组合
        for (loc1, loc2), count in location_pairs.items():
            if count >= 3:
                results.append(InferenceResult(
                    rule_id="location_pattern",
                    rule_name="地点组合模式",
                    result_type="pattern",
                    description=f"经常去 {loc1} 后也会去 {loc2}",
                    confidence=min(count / 5, 1.0),
                    suggested_action=f"在 {loc1} 时可以考虑顺便去 {loc2}"
                ))
        
        return results
    
    # ============ 辅助方法 ============
    
    def _calculate_distance(self, loc1: Dict, loc2: Dict) -> float:
        """计算两点距离（米）"""
        # 简化的距离计算
        lat_diff = loc1.get("lat", 0) - loc2.get("lat", 0)
        lng_diff = loc1.get("lng", 0) - loc2.get("lng", 0)
        return (lat_diff ** 2 + lng_diff ** 2) ** 0.5 * 111000  # 粗略转换
    
    def _hour_to_time_of_day(self, hour: int) -> str:
        """小时转换为时间段描述"""
        if 5 <= hour < 12:
            return "早晨"
        elif 12 <= hour < 14:
            return "中午"
        elif 14 <= hour < 18:
            return "下午"
        elif 18 <= hour < 22:
            return "晚上"
        else:
            return "深夜"
    
    def _find_related_concepts(self, entity_text: str) -> List[str]:
        """查找与实体文本相关的概念"""
        related = []
        for concept_id, concept in self.concepts.items():
            if entity_text in concept.label or concept.label in entity_text:
                related.append(concept_id)
                # 添加关系邻居
                for rel in concept.relations:
                    related.append(rel.get("target"))
        return list(set(related))


# 测试
if __name__ == "__main__":
    engine = LocalReasoningEngine()
    
    # 添加测试概念
    engine.add_concept(Concept("person_zhangsan", "PERSON", "张三"))
    engine.add_concept(Concept("person_lisi", "PERSON", "李四"))
    engine.add_concept(Concept("place_starbucks", "PLACE", "星巴克"))
    
    # 添加测试记忆
    engine.add_memory(Memory(
        id="mem1",
        content="今天早晨在星巴克和李四讨论项目",
        timestamp=datetime.now(),
        entities=[{"text": "李四", "label": "PERSON"}, {"text": "星巴克", "label": "PLACE"}],
        location={"lat": 39.9, "lng": 116.4, "name": "星巴克"},
        ontology_bindings=["person_lisi", "place_starbucks"]
    ))
    
    # 执行推理
    results = engine.infer()
    print(f"推理结果数量: {len(results)}")
    for r in results:
        print(f"- {r.rule_name}: {r.description}")
