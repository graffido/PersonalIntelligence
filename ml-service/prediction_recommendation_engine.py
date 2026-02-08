#!/usr/bin/env python3
"""
POS 预测与推荐系统
基于本体和记忆的模式进行智能推荐
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import math

class RecommendationType(Enum):
    TIME_BASED = "time_based"           # 基于时间的推荐
    LOCATION_BASED = "location_based"   # 基于位置的推荐
    SOCIAL_BASED = "social_based"       # 基于社交的推荐
    HABIT_BASED = "habit_based"         # 基于习惯的推荐
    PREDICTIVE = "predictive"           # 预测性推荐
    CONTEXTUAL = "contextual"           # 情境感知推荐

@dataclass
class Recommendation:
    """推荐项"""
    id: str
    type: RecommendationType
    title: str
    description: str
    confidence: float  # 0-1
    priority: int      # 1-5, 5最高
    
    # 上下文信息
    relevant_concepts: List[str] = field(default_factory=list)
    supporting_memories: List[str] = field(default_factory=list)
    
    # 行动建议
    suggested_action: Optional[str] = None
    suggested_time: Optional[datetime] = None
    suggested_location: Optional[Dict] = None
    
    # 元数据
    reason: str = ""  # 推荐理由
    expires_at: Optional[datetime] = None  # 推荐过期时间

@dataclass
class Prediction:
    """预测结果"""
    event_type: str
    predicted_time: datetime
    confidence: float
    location_probability: Dict[str, float]  # 地点 -> 概率
    person_probability: Dict[str, float]    # 人物 -> 概率
    explanation: str

class PredictionRecommendationEngine:
    """预测与推荐引擎"""
    
    def __init__(self, memory_store, ontology_graph, reasoning_engine):
        self.memory_store = memory_store
        self.ontology = ontology_graph
        self.reasoning = reasoning_engine
        
        # 习惯模式缓存
        self.habit_patterns: Dict[str, Dict] = {}
        
    def generate_recommendations(self, 
                                 current_context: Dict,
                                 limit: int = 5) -> List[Recommendation]:
        """
        生成推荐列表
        
        Args:
            current_context: 当前情境 (时间、位置、活动等)
        """
        recommendations = []
        
        # 1. 基于当前时间推荐
        time_recs = self._generate_time_based_recommendations(current_context)
        recommendations.extend(time_recs)
        
        # 2. 基于当前位置推荐
        if current_context.get('location'):
            loc_recs = self._generate_location_based_recommendations(current_context)
            recommendations.extend(loc_recs)
        
        # 3. 基于习惯模式推荐
        habit_recs = self._generate_habit_based_recommendations(current_context)
        recommendations.extend(habit_recs)
        
        # 4. 基于社交关系推荐
        social_recs = self._generate_social_recommendations(current_context)
        recommendations.extend(social_recs)
        
        # 5. 预测性推荐
        predictive_recs = self._generate_predictive_recommendations(current_context)
        recommendations.extend(predictive_recs)
        
        # 6. 冲突预警
        conflict_recs = self._generate_conflict_warnings(current_context)
        recommendations.extend(conflict_recs)
        
        # 排序：优先级 -> 置信度
        recommendations.sort(key=lambda r: (-r.priority, -r.confidence))
        
        return recommendations[:limit]
    
    def predict_next_events(self, 
                           from_time: datetime,
                           horizon_hours: int = 24) -> List[Prediction]:
        """预测未来事件"""
        predictions = []
        
        # 分析历史模式
        patterns = self._analyze_temporal_patterns()
        
        for pattern in patterns:
            if pattern['confidence'] > 0.6:
                # 计算下次发生时间
                next_time = self._calculate_next_occurrence(pattern, from_time)
                if next_time and next_time < from_time + timedelta(hours=horizon_hours):
                    pred = Prediction(
                        event_type=pattern['event_type'],
                        predicted_time=next_time,
                        confidence=pattern['confidence'],
                        location_probability=pattern.get('location_dist', {}),
                        person_probability=pattern.get('person_dist', {}),
                        explanation=pattern.get('explanation', '')
                    )
                    predictions.append(pred)
        
        return sorted(predictions, key=lambda p: p.predicted_time)
    
    def _generate_time_based_recommendations(self, context: Dict) -> List[Recommendation]:
        """基于时间的推荐"""
        recommendations = []
        current_time = context.get('current_time', datetime.now())
        hour = current_time.hour
        
        # 早晨推荐
        if 6 <= hour < 9:
            # 检查是否有早晨习惯
            morning_memories = self._get_memories_by_time_range(6, 9)
            if len(morning_memories) >= 3:
                common_activity = self._get_most_common_activity(morning_memories)
                recommendations.append(Recommendation(
                    id="morning_routine",
                    type=RecommendationType.TIME_BASED,
                    title="早晨习惯提醒",
                    description=f"您经常在早晨进行: {common_activity}",
                    confidence=min(len(morning_memories) / 10, 0.9),
                    priority=4,
                    reason="基于您的历史早晨活动模式",
                    suggested_time=current_time.replace(hour=8, minute=0)
                ))
        
        # 午餐时间推荐
        elif 11 <= hour < 13:
            lunch_places = self._get_frequent_places_by_time(11, 13)
            if lunch_places:
                place = lunch_places[0]
                recommendations.append(Recommendation(
                    id="lunch_suggestion",
                    type=RecommendationType.TIME_BASED,
                    title="午餐推荐",
                    description=f"您经常在午餐时间去: {place}",
                    confidence=0.75,
                    priority=3,
                    reason="基于您的午餐习惯",
                    suggested_action=f"可以考虑去 {place} 用餐"
                ))
        
        # 晚上推荐
        elif 18 <= hour < 22:
            evening_memories = self._get_memories_by_time_range(18, 22)
            if len(evening_memories) >= 5:
                # 检查是否是锻炼时间
                exercise_count = sum(1 for m in evening_memories if '健身' in m or '锻炼' in m)
                if exercise_count >= 3:
                    recommendations.append(Recommendation(
                        id="evening_exercise",
                        type=RecommendationType.HABIT_BASED,
                        title="锻炼提醒",
                        description="您经常在晚上锻炼，今天也要去吗？",
                        confidence=exercise_count / len(evening_memories),
                        priority=4,
                        reason="基于您的晚间锻炼习惯",
                        suggested_action="前往常去的健身房"
                    ))
        
        return recommendations
    
    def _generate_location_based_recommendations(self, context: Dict) -> List[Recommendation]:
        """基于位置的推荐"""
        recommendations = []
        current_location = context.get('location', {})
        
        if not current_location:
            return recommendations
        
        # 查找附近相关的记忆
        nearby_memories = self._get_memories_near_location(
            current_location.get('lat'),
            current_location.get('lng'),
            radius_meters=500
        )
        
        if nearby_memories:
            # 提取相关人物
            related_persons = self._extract_persons_from_memories(nearby_memories)
            if related_persons:
                recommendations.append(Recommendation(
                    id="nearby_people",
                    type=RecommendationType.LOCATION_BASED,
                    title="附近联系人",
                    description=f"您曾在这里与 {', '.join(related_persons[:3])} 见面",
                    confidence=0.7,
                    priority=3,
                    reason="基于该地点的历史社交记录",
                    suggested_action=f"如果需要，可以联系 {related_persons[0]}"
                ))
            
            # 推荐附近其他常去地点
            frequent_nearby = self._get_frequent_nearby_places(current_location)
            if frequent_nearby:
                recommendations.append(Recommendation(
                    id="nearby_places",
                    type=RecommendationType.LOCATION_BASED,
                    title="附近推荐",
                    description=f"距离这里不远处有您常去的: {frequent_nearby[0]}",
                    confidence=0.6,
                    priority=2,
                    reason="基于位置关联模式"
                ))
        
        return recommendations
    
    def _generate_habit_based_recommendations(self, context: Dict) -> List[Recommendation]:
        """基于习惯的推荐"""
        recommendations = []
        
        # 分析周期性模式
        patterns = self._analyze_temporal_patterns()
        
        for pattern in patterns:
            if pattern['confidence'] > 0.7 and pattern['frequency'] >= 4:
                # 检查是否是今天
                if self._is_pattern_due_today(pattern):
                    recommendations.append(Recommendation(
                        id=f"habit_{pattern['id']}",
                        type=RecommendationType.HABIT_BASED,
                        title="习惯提醒",
                        description=pattern['description'],
                        confidence=pattern['confidence'],
                        priority=4,
                        reason=f"您已连续{pattern['frequency']}次在此时进行此活动",
                        suggested_action=pattern.get('suggested_action', ''),
                        suggested_time=pattern.get('usual_time')
                    ))
        
        return recommendations
    
    def _generate_social_recommendations(self, context: Dict) -> List[Recommendation]:
        """基于社交的推荐"""
        recommendations = []
        
        # 计算关系强度
        relationships = self._calculate_relationship_strengths()
        
        # 对关系较弱但曾频繁互动的人提醒
        for person, strength in relationships.items():
            if 0.3 < strength < 0.6:
                days_since_last = self._days_since_last_interaction(person)
                if days_since_last > 30:
                    recommendations.append(Recommendation(
                        id=f"social_{person}",
                        type=RecommendationType.SOCIAL_BASED,
                        title="社交提醒",
                        description=f"您已经 {days_since_last} 天没有和 {person} 联系了",
                        confidence=min(days_since_last / 60, 0.8),
                        priority=3,
                        reason="基于社交关系维护",
                        suggested_action=f"可以考虑联系 {person}"
                    ))
        
        # 朋友的朋友推荐
        friend_suggestions = self._get_friend_of_friend_suggestions()
        for suggestion in friend_suggestions[:2]:
            recommendations.append(Recommendation(
                id=f"fof_{suggestion['person']}",
                type=RecommendationType.SOCIAL_BASED,
                title="潜在联系人",
                description=f"{suggestion['via']} 的朋友 {suggestion['person']} 可能也认识您",
                confidence=suggestion['confidence'],
                priority=2,
                reason="基于社交图谱传递性"
            ))
        
        return recommendations
    
    def _generate_predictive_recommendations(self, context: Dict) -> List[Recommendation]:
        """预测性推荐"""
        recommendations = []
        
        # 预测接下来的活动
        predictions = self.predict_next_events(
            datetime.now(),
            horizon_hours=48
        )
        
        for pred in predictions[:3]:
            # 找到最可能的地点和人物
            top_location = max(pred.location_probability.items(), key=lambda x: x[1])
            top_person = max(pred.person_probability.items(), key=lambda x: x[1]) if pred.person_probability else None
            
            rec = Recommendation(
                id=f"predict_{pred.event_type}",
                type=RecommendationType.PREDICTIVE,
                title="活动预测",
                description=f"预计您在 {pred.predicted_time.strftime('%m月%d日 %H:%M')} 可能会: {pred.event_type}",
                confidence=pred.confidence,
                priority=3 if pred.confidence > 0.7 else 2,
                reason=pred.explanation,
                suggested_time=pred.predicted_time,
                suggested_location={"name": top_location[0]} if top_location[1] > 0.5 else None
            )
            
            if top_person and top_person[1] > 0.5:
                rec.description += f" (可能和 {top_person[0]} 一起)"
            
            recommendations.append(rec)
        
        return recommendations
    
    def _generate_conflict_warnings(self, context: Dict) -> List[Recommendation]:
        """生成冲突预警"""
        recommendations = []
        
        # 检查未来24小时的日程
        upcoming = self._get_upcoming_memories(hours=24)
        
        # 检查时间冲突
        for i, mem1 in enumerate(upcoming):
            for mem2 in upcoming[i+1:]:
                if self._have_time_conflict(mem1, mem2):
                    recommendations.append(Recommendation(
                        id=f"conflict_{mem1['id']}_{mem2['id']}",
                        type=RecommendationType.CONTEXTUAL,
                        title="⚠️ 日程冲突",
                        description=f"时间冲突: {mem1['title']} 和 {mem2['title']}",
                        confidence=0.95,
                        priority=5,
                        reason="检测到时间重叠",
                        suggested_action="请重新安排其中一个日程",
                        expires_at=mem1['time']
                    ))
        
        # 检查可达性问题
        for mem in upcoming:
            if mem.get('location') and context.get('current_location'):
                travel_time = self._estimate_travel_time(
                    context['current_location'],
                    mem['location']
                )
                time_until = (mem['time'] - datetime.now()).total_seconds() / 60
                
                if travel_time > time_until - 15:  # 需要提前15分钟到达
                    recommendations.append(Recommendation(
                        id=f"travel_{mem['id']}",
                        type=RecommendationType.CONTEXTUAL,
                        title="⏰ 出行提醒",
                        description=f"距离 {mem['title']} 还有 {int(time_until)} 分钟，但路上需要 {int(travel_time)} 分钟",
                        confidence=0.85,
                        priority=5,
                        reason="基于当前位置和交通时间估算",
                        suggested_action="建议立即出发",
                        expires_at=mem['time']
                    ))
        
        return recommendations
    
    # ============ 辅助方法 ============
    
    def _get_memories_by_time_range(self, start_hour: int, end_hour: int) -> List[Dict]:
        """获取特定时间段的记忆"""
        # 实际应从memory_store查询
        return []
    
    def _get_most_common_activity(self, memories: List[Dict]) -> str:
        """获取最常见的活动"""
        activities = {}
        for m in memories:
            act = m.get('activity', '其他')
            activities[act] = activities.get(act, 0) + 1
        return max(activities.items(), key=lambda x: x[1])[0] if activities else ""
    
    def _get_frequent_places_by_time(self, start_hour: int, end_hour: int) -> List[str]:
        """获取特定时段常去的地点"""
        return []
    
    def _get_memories_near_location(self, lat: float, lng: float, radius_meters: float) -> List[Dict]:
        """获取位置附近的记忆"""
        return []
    
    def _extract_persons_from_memories(self, memories: List[Dict]) -> List[str]:
        """从记忆中提取人物"""
        persons = set()
        for m in memories:
            for p in m.get('persons', []):
                persons.add(p)
        return list(persons)
    
    def _get_frequent_nearby_places(self, location: Dict) -> List[str]:
        """获取附近常去的地方"""
        return []
    
    def _analyze_temporal_patterns(self) -> List[Dict]:
        """分析时间模式"""
        return []
    
    def _is_pattern_due_today(self, pattern: Dict) -> bool:
        """检查模式是否今天到期"""
        return False
    
    def _calculate_next_occurrence(self, pattern: Dict, from_time: datetime) -> Optional[datetime]:
        """计算下次发生时间"""
        return None
    
    def _calculate_relationship_strengths(self) -> Dict[str, float]:
        """计算关系强度"""
        return {}
    
    def _days_since_last_interaction(self, person: str) -> int:
        """计算上次互动距今天数"""
        return 0
    
    def _get_friend_of_friend_suggestions(self) -> List[Dict]:
        """获取朋友的朋友建议"""
        return []
    
    def _get_upcoming_memories(self, hours: int) -> List[Dict]:
        """获取即将到来的记忆/日程"""
        return []
    
    def _have_time_conflict(self, mem1: Dict, mem2: Dict) -> bool:
        """检查是否有时间冲突"""
        return False
    
    def _estimate_travel_time(self, from_loc: Dict, to_loc: Dict) -> float:
        """估算交通时间（分钟）"""
        # 简化的距离计算
        dist = math.sqrt(
            (from_loc.get('lat', 0) - to_loc.get('lat', 0)) ** 2 +
            (from_loc.get('lng', 0) - to_loc.get('lng', 0)) ** 2
        ) * 111  # 转换为km
        # 假设平均速度30km/h
        return (dist / 30) * 60


# 演示
if __name__ == "__main__":
    engine = PredictionRecommendationEngine(None, None, None)
    
    # 模拟当前情境
    context = {
        'current_time': datetime.now().replace(hour=8, minute=30),
        'location': {'lat': 39.9, 'lng': 116.4, 'name': '家'},
        'activity': '起床'
    }
    
    print("=" * 60)
    print("预测与推荐系统演示")
    print("=" * 60)
    print(f"\n当前情境: {context['current_time'].strftime('%Y-%m-%d %H:%M')}")
    print(f"位置: {context['location']['name']}")
    
    # 这里需要实际的memory_store数据才能生成真实推荐
    print("\n(需要连接实际数据存储才能生成完整推荐)")
