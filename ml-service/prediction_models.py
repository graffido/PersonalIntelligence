"""
预测模型模块
实现时序预测、地点预测、人物预测
"""

import json
import math
import random
from collections import defaultdict, Counter
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Any, Callable
import logging

logger = logging.getLogger(__name__)

# ============== 基础数据模型 ==============

@dataclass
class TimeSeriesPoint:
    """时序数据点"""
    timestamp: datetime
    value: float
    features: Optional[Dict[str, Any]] = None

@dataclass
class PredictionResult:
    """预测结果"""
    predicted_value: Any
    confidence: float
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None
    explanation: str = ""

@dataclass
class LocationPrediction:
    """地点预测结果"""
    location: str
    probability: float
    expected_time: Optional[datetime] = None

@dataclass
class PersonPrediction:
    """人物预测结果"""
    person_id: str
    person_name: str
    meet_probability: float
    expected_meet_time: Optional[datetime] = None
    context: str = ""


# ============== 1. 时序预测 (Prophet-like + LSTM-like) ==============

class TimeSeriesPredictor:
    """
    时序预测器
    结合趋势分解、季节性和简单的神经网络思想
    """
    
    def __init__(self, seasonality_period: int = 24):
        """
        Args:
            seasonality_period: 季节性周期（小时为单位，默认24小时）
        """
        self.seasonality_period = seasonality_period
        self.trend_params: Optional[Tuple[float, float]] = None  # (slope, intercept)
        self.seasonal_pattern: Dict[int, float] = {}
        self.residuals: List[float] = []
        self.history: List[TimeSeriesPoint] = []
    
    def fit(self, data: List[TimeSeriesPoint]):
        """
        拟合时序模型
        
        使用简单的加法分解: Y = Trend + Seasonal + Residual
        """
        if len(data) < self.seasonality_period * 2:
            logger.warning("数据量不足，可能无法准确捕捉季节性")
        
        self.history = sorted(data, key=lambda x: x.timestamp)
        values = [p.value for p in self.history]
        
        # 1. 提取趋势 (简单线性回归)
        self.trend_params = self._fit_trend(values)
        
        # 2. 去趋势
        detrended = self._remove_trend(values)
        
        # 3. 提取季节性模式
        self.seasonal_pattern = self._extract_seasonality(detrended)
        
        # 4. 计算残差
        self.residuals = self._calculate_residuals(values)
        
        logger.info(f"时序模型拟合完成: 趋势参数={self.trend_params}")
    
    def _fit_trend(self, values: List[float]) -> Tuple[float, float]:
        """使用最小二乘法拟合线性趋势"""
        n = len(values)
        if n < 2:
            return (0.0, values[0] if values else 0.0)
        
        x = list(range(n))
        x_mean = sum(x) / n
        y_mean = sum(values) / n
        
        # 计算斜率和截距
        numerator = sum((x[i] - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
        
        slope = numerator / denominator if denominator != 0 else 0
        intercept = y_mean - slope * x_mean
        
        return (slope, intercept)
    
    def _remove_trend(self, values: List[float]) -> List[float]:
        """从数据中移除趋势"""
        slope, intercept = self.trend_params
        return [values[i] - (intercept + slope * i) for i in range(len(values))]
    
    def _extract_seasonality(self, detrended: List[float]) -> Dict[int, float]:
        """提取季节性模式"""
        seasonal_sums = defaultdict(list)
        
        for i, value in enumerate(detrended):
            period_idx = i % self.seasonality_period
            seasonal_sums[period_idx].append(value)
        
        # 计算每个季节位置的平均值
        return {
            idx: sum(values) / len(values) 
            for idx, values in seasonal_sums.items()
        }
    
    def _calculate_residuals(self, values: List[float]) -> List[float]:
        """计算残差"""
        slope, intercept = self.trend_params
        residuals = []
        
        for i, value in enumerate(values):
            trend = intercept + slope * i
            seasonal = self.seasonal_pattern.get(i % self.seasonality_period, 0)
            residual = value - trend - seasonal
            residuals.append(residual)
        
        return residuals
    
    def predict(self, steps: int = 1, confidence: float = 0.95) -> List[PredictionResult]:
        """
        预测未来值
        
        Args:
            steps: 预测步数
            confidence: 置信区间概率
        """
        if not self.history or self.trend_params is None:
            return []
        
        results = []
        slope, intercept = self.trend_params
        n = len(self.history)
        
        # 计算残差标准差用于置信区间
        residual_std = (sum(r ** 2 for r in self.residuals) / len(self.residuals)) ** 0.5
        z_score = 1.96 if confidence == 0.95 else 1.645  # 简化处理
        
        for step in range(1, steps + 1):
            future_idx = n + step - 1
            
            # 趋势预测
            trend = intercept + slope * future_idx
            
            # 季节性预测
            seasonal = self.seasonal_pattern.get(future_idx % self.seasonality_period, 0)
            
            # 组合预测
            predicted = trend + seasonal
            
            # 置信区间（随着时间步增加而变宽）
            margin = z_score * residual_std * (1 + 0.1 * step)
            
            results.append(PredictionResult(
                predicted_value=predicted,
                confidence=confidence,
                lower_bound=predicted - margin,
                upper_bound=predicted + margin,
                explanation=f"趋势={trend:.2f}, 季节性={seasonal:.2f}"
            ))
        
        return results
    
    def predict_next_timestamp(self, last_timestamp: datetime, 
                               steps: int = 1) -> List[Tuple[datetime, PredictionResult]]:
        """预测特定时间点的值"""
        predictions = self.predict(steps)
        results = []
        
        # 假设数据点是每小时一个
        for i, pred in enumerate(predictions):
            future_time = last_timestamp + timedelta(hours=i+1)
            results.append((future_time, pred))
        
        return results
    
    def detect_anomalies(self, threshold: float = 2.0) -> List[Tuple[int, TimeSeriesPoint, float]]:
        """
        检测异常值
        
        Returns:
            [(index, data_point, anomaly_score), ...]
        """
        if not self.residuals:
            return []
        
        residual_std = (sum(r ** 2 for r in self.residuals) / len(self.residuals)) ** 0.5
        anomalies = []
        
        for i, residual in enumerate(self.residuals):
            score = abs(residual) / residual_std
            if score > threshold:
                anomalies.append((i, self.history[i], score))
        
        return anomalies


class LSTMPredictor:
    """
    简化版LSTM预测器
    使用滑动窗口和简单的RNN思想
    """
    
    def __init__(self, window_size: int = 10, hidden_size: int = 20):
        self.window_size = window_size
        self.hidden_size = hidden_size
        self.weights: Optional[List[List[float]]] = None
        self.history: List[float] = []
    
    def fit(self, data: List[TimeSeriesPoint]):
        """拟合模型（简化版使用移动平均）"""
        self.history = [p.value for p in sorted(data, key=lambda x: x.timestamp)]
        
        # 简化：计算各窗口的权重模式
        if len(self.history) < self.window_size * 2:
            return
        
        # 计算每个位置的平均影响权重
        weights = []
        for lag in range(1, self.window_size + 1):
            correlations = []
            for i in range(lag, len(self.history)):
                correlations.append(self.history[i] * self.history[i - lag])
            avg_corr = sum(correlations) / len(correlations) if correlations else 0
            weights.append(avg_corr)
        
        # 归一化权重
        total = sum(abs(w) for w in weights)
        if total > 0:
            self.weights = [[w / total] for w in weights]
        
        logger.info("LSTM-like模型拟合完成")
    
    def predict(self, steps: int = 1) -> List[PredictionResult]:
        """预测未来值"""
        if not self.history or self.weights is None:
            return []
        
        predictions = []
        temp_history = self.history.copy()
        
        for _ in range(steps):
            # 取最近的window_size个值
            window = temp_history[-self.window_size:]
            if len(window) < self.window_size:
                window = [0.0] * (self.window_size - len(window)) + window
            
            # 加权预测
            predicted = sum(
                window[-i-1] * self.weights[i][0] 
                for i in range(len(self.weights))
            )
            
            predictions.append(PredictionResult(
                predicted_value=predicted,
                confidence=0.8,  # 简化
                explanation="基于滑动窗口的预测"
            ))
            
            temp_history.append(predicted)
        
        return predictions


# ============== 2. 地点预测 (马尔可夫链) ==============

class LocationPredictor:
    """
    地点预测器
    使用马尔可夫链建模地点转移概率
    """
    
    def __init__(self, order: int = 1):
        """
        Args:
            order: 马尔可夫链阶数（1阶表示仅依赖前一个地点）
        """
        self.order = order
        self.transitions: Dict[Tuple[str, ...], Counter] = defaultdict(Counter)
        self.location_counts: Counter = Counter()
        self.location_times: Dict[str, List[datetime]] = defaultdict(list)
        self.all_locations: Set[str] = set()
    
    def fit(self, location_history: List[Tuple[datetime, str]]):
        """
        从地点历史中学习转移概率
        
        Args:
            location_history: [(timestamp, location), ...]
        """
        # 按时间排序
        sorted_history = sorted(location_history, key=lambda x: x[0])
        locations = [loc for _, loc in sorted_history]
        
        self.all_locations = set(locations)
        
        # 统计转移
        for i in range(len(locations) - self.order):
            current_state = tuple(locations[i:i + self.order])
            next_location = locations[i + self.order]
            
            self.transitions[current_state][next_location] += 1
            self.location_counts[next_location] += 1
            self.location_times[next_location].append(sorted_history[i + self.order][0])
        
        logger.info(f"地点转移模型训练完成: {len(self.transitions)} 个状态")
    
    def predict_next(self, recent_locations: List[str], 
                     top_k: int = 3) -> List[LocationPrediction]:
        """
        预测下一个地点
        
        Args:
            recent_locations: 最近的地点序列（长度应等于order）
            top_k: 返回前k个最可能的地点
        """
        if len(recent_locations) < self.order:
            # 返回最频繁的地点
            most_common = self.location_counts.most_common(top_k)
            return [
                LocationPrediction(
                    location=loc,
                    probability=count / sum(self.location_counts.values()),
                    expected_time=self._predict_arrival_time(loc)
                )
                for loc, count in most_common
            ]
        
        # 获取当前状态
        current_state = tuple(recent_locations[-self.order:])
        
        # 获取转移概率
        next_locations = self.transitions.get(current_state, Counter())
        
        if not next_locations:
            # 回退到全局频率
            most_common = self.location_counts.most_common(top_k)
            total = sum(self.location_counts.values())
            return [
                LocationPrediction(
                    location=loc,
                    probability=count / total,
                    expected_time=self._predict_arrival_time(loc)
                )
                for loc, count in most_common
            ]
        
        # 计算概率
        total = sum(next_locations.values())
        sorted_locations = next_locations.most_common(top_k)
        
        return [
            LocationPrediction(
                location=loc,
                probability=count / total,
                expected_time=self._predict_arrival_time(loc)
            )
            for loc, count in sorted_locations
        ]
    
    def _predict_arrival_time(self, location: str) -> Optional[datetime]:
        """预测到达某地点的时间"""
        times = self.location_times.get(location, [])
        if not times:
            return None
        
        # 计算平均间隔
        if len(times) < 2:
            return None
        
        intervals = [
            (times[i] - times[i-1]).total_seconds() / 3600  # 小时
            for i in range(1, len(times))
        ]
        avg_interval = sum(intervals) / len(intervals)
        
        # 预测下一次到达时间
        last_visit = max(times)
        return last_visit + timedelta(hours=avg_interval)
    
    def predict_sequence(self, start_locations: List[str], 
                        steps: int = 3) -> List[List[LocationPrediction]]:
        """预测未来地点序列"""
        sequence = []
        current = start_locations.copy()
        
        for _ in range(steps):
            predictions = self.predict_next(current, top_k=1)
            sequence.append(predictions)
            
            if predictions:
                current.append(predictions[0].location)
                if len(current) > self.order:
                    current = current[-self.order:]
            else:
                break
        
        return sequence
    
    def get_location_stats(self) -> Dict[str, Dict]:
        """获取地点统计信息"""
        stats = {}
        total_visits = sum(self.location_counts.values())
        
        for location in self.all_locations:
            times = self.location_times[location]
            
            # 计算平均停留时间（简化：假设在下一个地点出现前）
            avg_interval = None
            if len(times) >= 2:
                intervals = [
                    (times[i] - times[i-1]).total_seconds() / 3600
                    for i in range(1, len(times))
                ]
                avg_interval = sum(intervals) / len(intervals)
            
            # 计算访问频率（每周）
            if times:
                time_span = (max(times) - min(times)).total_seconds() / 86400  # 天数
                weekly_freq = len(times) / (time_span / 7) if time_span > 0 else 0
            else:
                weekly_freq = 0
            
            stats[location] = {
                "visit_count": self.location_counts[location],
                "visit_probability": self.location_counts[location] / total_visits,
                "avg_interval_hours": avg_interval,
                "weekly_frequency": weekly_freq,
                "first_visit": min(times) if times else None,
                "last_visit": max(times) if times else None
            }
        
        return stats


# ============== 3. 人物预测 (社交关系演化) ==============

class PersonPredictor:
    """
    人物预测器
    预测与谁可能见面、何时见面
    """
    
    def __init__(self):
        self.interaction_history: List[Dict] = []
        self.person_stats: Dict[str, Dict] = defaultdict(lambda: {
            'interaction_count': 0,
            'last_meeting': None,
            'meetings': [],
            'contexts': Counter()
        })
        self.context_persons: Dict[str, Counter] = defaultdict(Counter)
        self.time_patterns: Dict[str, List[int]] = defaultdict(list)  # 小时分布
    
    def fit(self, interactions: List[Dict]):
        """
        从交互历史中学习
        
        Args:
            interactions: [
                {
                    'person_id': str,
                    'person_name': str,
                    'timestamp': datetime,
                    'context': str,  # 如 'meeting', 'lunch', 'call'
                    'location': str,
                    'duration_minutes': int
                },
                ...
            ]
        """
        self.interaction_history = sorted(
            interactions, 
            key=lambda x: x.get('timestamp', datetime.min)
        )
        
        for interaction in self.interaction_history:
            person_id = interaction.get('person_id')
            if not person_id:
                continue
            
            stats = self.person_stats[person_id]
            stats['interaction_count'] += 1
            stats['meetings'].append(interaction)
            
            timestamp = interaction.get('timestamp')
            if timestamp:
                stats['last_meeting'] = timestamp
                self.time_patterns[person_id].append(timestamp.hour)
            
            context = interaction.get('context', 'unknown')
            stats['contexts'][context] += 1
            self.context_persons[context][person_id] += 1
        
        logger.info(f"人物关系模型训练完成: {len(self.person_stats)} 个人")
    
    def predict_next_meetings(self, context: str = None, 
                              top_k: int = 5) -> List[PersonPrediction]:
        """
        预测下一次最可能见面的人
        
        Args:
            context: 特定场景（如'meeting', 'lunch'）
            top_k: 返回前k个最可能的人
        """
        predictions = []
        
        for person_id, stats in self.person_stats.items():
            # 计算基础概率
            total_interactions = sum(s['interaction_count'] for s in self.person_stats.values())
            base_prob = stats['interaction_count'] / total_interactions if total_interactions > 0 else 0
            
            # 考虑时间衰减
            if stats['last_meeting']:
                days_since = (datetime.now() - stats['last_meeting']).days
                time_decay = math.exp(-days_since / 30)  # 30天衰减因子
            else:
                time_decay = 1.0
            
            # 考虑上下文匹配
            context_boost = 1.0
            if context and context in stats['contexts']:
                context_ratio = stats['contexts'][context] / stats['interaction_count']
                context_boost = 1 + context_ratio
            
            # 综合概率
            probability = base_prob * time_decay * context_boost
            
            # 预测见面时间
            expected_time = self._predict_meeting_time(person_id)
            
            # 生成上下文描述
            ctx_desc = self._generate_context_description(stats, context)
            
            predictions.append(PersonPrediction(
                person_id=person_id,
                person_name=stats['meetings'][0].get('person_name', person_id) if stats['meetings'] else person_id,
                meet_probability=probability,
                expected_meet_time=expected_time,
                context=ctx_desc
            ))
        
        # 排序并返回前k个
        predictions.sort(key=lambda x: x.meet_probability, reverse=True)
        return predictions[:top_k]
    
    def _predict_meeting_time(self, person_id: str) -> Optional[datetime]:
        """预测与某人见面的时间"""
        stats = self.person_stats.get(person_id)
        if not stats or not stats['last_meeting']:
            return None
        
        meetings = stats['meetings']
        if len(meetings) < 2:
            return None
        
        # 计算平均间隔
        intervals = []
        for i in range(1, len(meetings)):
            t1 = meetings[i-1].get('timestamp')
            t2 = meetings[i].get('timestamp')
            if t1 and t2:
                intervals.append((t2 - t1).total_seconds() / 86400)  # 天数
        
        if not intervals:
            return None
        
        avg_interval = sum(intervals) / len(intervals)
        
        # 预测下一次见面时间
        return stats['last_meeting'] + timedelta(days=avg_interval)
    
    def _generate_context_description(self, stats: Dict, context: str = None) -> str:
        """生成上下文描述"""
        if context:
            count = stats['contexts'].get(context, 0)
            if count > 0:
                return f"经常在{context}场景下见面（{count}次）"
        
        # 找出最常见的上下文
        if stats['contexts']:
            most_common = stats['contexts'].most_common(1)[0]
            return f"通常在{most_common[0]}场景下见面"
        
        return ""
    
    def predict_for_context(self, context: str, top_k: int = 5) -> List[PersonPrediction]:
        """预测在特定场景下会见到的人"""
        persons = self.context_persons.get(context, Counter())
        
        predictions = []
        for person_id, count in persons.most_common(top_k):
            stats = self.person_stats[person_id]
            total_context = sum(self.context_persons[context].values())
            
            predictions.append(PersonPrediction(
                person_id=person_id,
                person_name=stats['meetings'][0].get('person_name', person_id) if stats['meetings'] else person_id,
                meet_probability=count / total_context if total_context > 0 else 0,
                expected_meet_time=self._predict_meeting_time(person_id),
                context=f"在{context}场景下见面{count}次"
            ))
        
        return predictions
    
    def get_relationship_evolution(self, person_id: str) -> List[Dict]:
        """
        获取与某人的关系演化历史
        
        Returns:
            按时间排序的交互历史
        """
        stats = self.person_stats.get(person_id)
        if not stats:
            return []
        
        return [
            {
                'timestamp': m.get('timestamp'),
                'context': m.get('context'),
                'location': m.get('location'),
                'duration': m.get('duration_minutes')
            }
            for m in stats['meetings']
        ]
    
    def get_network_insights(self) -> Dict:
        """获取社交网络洞察"""
        insights = {
            'total_people': len(self.person_stats),
            'total_interactions': sum(
                s['interaction_count'] for s in self.person_stats.values()
            ),
            'top_interactions': [],
            'interaction_trends': []
        }
        
        # 找出交互最频繁的人
        sorted_people = sorted(
            self.person_stats.items(),
            key=lambda x: x[1]['interaction_count'],
            reverse=True
        )[:10]
        
        insights['top_interactions'] = [
            {
                'person_id': pid,
                'count': stats['interaction_count'],
                'last_meeting': stats['last_meeting'],
                'top_context': stats['contexts'].most_common(1)[0] if stats['contexts'] else None
            }
            for pid, stats in sorted_people
        ]
        
        return insights


# ============== 综合预测引擎 ==============

class PredictionEngine:
    """综合预测引擎"""
    
    def __init__(self):
        self.time_predictor = TimeSeriesPredictor()
        self.location_predictor = LocationPredictor()
        self.person_predictor = PersonPredictor()
        
        self.predictions_history: List[Dict] = []
    
    def train_time_series(self, data: List[TimeSeriesPoint]):
        """训练时序预测模型"""
        self.time_predictor.fit(data)
    
    def train_location(self, location_history: List[Tuple[datetime, str]]):
        """训练地点预测模型"""
        self.location_predictor.fit(location_history)
    
    def train_person(self, interactions: List[Dict]):
        """训练人物预测模型"""
        self.person_predictor.fit(interactions)
    
    def predict_daily_routine(self, date: datetime) -> Dict:
        """
        预测某一天的日程
        
        Returns:
            包含地点、人物预测的日程
        """
        routine = {
            'date': date.strftime('%Y-%m-%d'),
            'predictions': []
        }
        
        # 预测全天各个时段的地点
        for hour in [9, 12, 15, 18, 21]:
            time_point = date.replace(hour=hour, minute=0)
            
            # 这里简化处理，实际应该基于历史该时段的地点
            location_preds = self.location_predictor.predict_next(['家'], top_k=1)
            person_preds = self.person_predictor.predict_next_meetings(top_k=2)
            
            routine['predictions'].append({
                'time': time_point.strftime('%H:%M'),
                'location': location_preds[0].location if location_preds else None,
                'people': [p.person_name for p in person_preds[:2]],
                'confidence': location_preds[0].probability if location_preds else 0
            })
        
        return routine
    
    def get_comprehensive_insights(self) -> Dict:
        """获取综合洞察"""
        return {
            'location_insights': self.location_predictor.get_location_stats(),
            'person_insights': self.person_predictor.get_network_insights(),
            'anomalies': self.time_predictor.detect_anomalies() if self.time_predictor.history else []
        }


# ============== 使用示例 ==============

def example_usage():
    """使用示例"""
    engine = PredictionEngine()
    
    # 示例1: 时序预测
    print("=== 时序预测示例 ===")
    
    # 生成示例时序数据（模拟每天的活动量）
    base_time = datetime.now() - timedelta(days=30)
    time_series_data = []
    
    for i in range(30):
        # 模拟周期性和趋势
        trend = i * 0.5
        seasonal = 10 * math.sin(i * 2 * math.pi / 7)  # 周周期
        noise = random.gauss(0, 2)
        value = 50 + trend + seasonal + noise
        
        time_series_data.append(TimeSeriesPoint(
            timestamp=base_time + timedelta(days=i),
            value=value
        ))
    
    engine.train_time_series(time_series_data)
    
    predictions = engine.time_predictor.predict(steps=3)
    for i, pred in enumerate(predictions):
        print(f"Day {i+1}: 预测值={pred.predicted_value:.2f}, "
              f"置信区间=[{pred.lower_bound:.2f}, {pred.upper_bound:.2f}]")
    
    # 示例2: 地点预测
    print("\n=== 地点预测示例 ===")
    
    location_history = [
        (base_time + timedelta(hours=i*8), loc)
        for i, loc in enumerate([
            '家', '办公室', '家', '办公室', '家',
            '家', '健身房', '家', '办公室', '咖啡厅',
            '家', '办公室', '家', '餐厅', '家'
        ])
    ]
    
    engine.train_location(location_history)
    
    next_locations = engine.location_predictor.predict_next(['办公室'], top_k=3)
    print("从办公室离开后最可能去的地方:")
    for loc in next_locations:
        print(f"  {loc.location}: 概率={loc.probability:.2%}")
    
    # 示例3: 人物预测
    print("\n=== 人物预测示例 ===")
    
    interactions = [
        {'person_id': 'p1', 'person_name': '张三', 'timestamp': base_time + timedelta(days=i*2),
         'context': '会议', 'location': '办公室', 'duration_minutes': 60}
        for i in range(5)
    ] + [
        {'person_id': 'p2', 'person_name': '李四', 'timestamp': base_time + timedelta(days=i*3+1),
         'context': '午餐', 'location': '餐厅', 'duration_minutes': 45}
        for i in range(4)
    ] + [
        {'person_id': 'p3', 'person_name': '王五', 'timestamp': base_time + timedelta(days=i*5),
         'context': '电话', 'location': '家', 'duration_minutes': 30}
        for i in range(3)
    ]
    
    engine.train_person(interactions)
    
    person_preds = engine.person_predictor.predict_next_meetings(top_k=3)
    print("最可能见面的人:")
    for p in person_preds:
        print(f"  {p.person_name}: 概率={p.meet_probability:.2%}, {p.context}")
    
    # 示例4: 综合预测
    print("\n=== 综合日程预测示例 ===")
    routine = engine.predict_daily_routine(datetime.now())
    print(f"日期: {routine['date']}")
    for pred in routine['predictions']:
        print(f"  {pred['time']}: 地点={pred['location']}, "
              f"可能见到={', '.join(pred['people'])}, 置信度={pred['confidence']:.2%}")


if __name__ == "__main__":
    example_usage()
