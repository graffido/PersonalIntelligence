#!/usr/bin/env python3
"""
POS 统一输入处理管道
单一输入框 -> 自动提取所有信息 -> 去重消歧 -> 存储
"""

from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime
import re
import hashlib
from difflib import SequenceMatcher

@dataclass
class ExtractedEntity:
    """提取的实体"""
    text: str
    label: str  # PERSON, PLACE, TIME, EVENT, OBJECT, ORGANIZATION
    start: int
    end: int
    confidence: float
    normalized_form: Optional[str] = None  # 规范化形式
    canonical_id: Optional[str] = None     # 指向标准概念的ID

@dataclass
class ExtractedRelation:
    """提取的关系"""
    subject: str
    predicate: str
    object: str
    confidence: float
    temporal_constraint: Optional[str] = None
    spatial_constraint: Optional[str] = None

@dataclass
class ParsedInput:
    """解析后的输入"""
    raw_text: str
    entities: List[ExtractedEntity]
    relations: List[ExtractedRelation]
    timestamp: Optional[datetime] = None
    location: Optional[Dict] = None
    main_event: Optional[str] = None
    sentiment: Optional[str] = None
    embedding: Optional[List[float]] = None

class UnifiedInputParser:
    """统一输入解析器"""
    
    def __init__(self):
        # 实体别名映射 (用于消歧)
        self.entity_aliases = {
            # 人名变体
            "中伟": ["中伟", "张伟", "张中伟", "伟哥"],
            "李四": ["李四", "老李", "四哥", "Li Si"],
            # 地点变体
            "星巴克": ["星巴克", "Starbucks", "SB", "星爸爸"],
            "海底捞": ["海底捞", "HDL", "捞海底"],
            # 时间变体
            "今天": ["今天", "今日", "这天"],
            "明天": ["明天", "明日"],
        }
        
        # 已知的标准概念 (用于去重)
        self.canonical_entities: Dict[str, Dict] = {}
        
    def parse(self, text: str, context: Optional[Dict] = None) -> ParsedInput:
        """
        主解析函数 - 单一入口
        
        Args:
            text: 用户输入的自然语言
            context: 可选的上下文 (当前时间、位置等)
        """
        # 1. 实体提取
        entities = self._extract_entities(text)
        
        # 2. 实体消歧与规范化
        entities = self._disambiguate_entities(entities)
        
        # 3. 去重
        entities = self._deduplicate_entities(entities)
        
        # 4. 关系抽取
        relations = self._extract_relations(text, entities)
        
        # 5. 时间解析
        timestamp = self._parse_time(text, context)
        
        # 6. 地点解析
        location = self._parse_location(text, entities)
        
        # 7. 事件识别
        main_event = self._identify_main_event(text, entities)
        
        # 8. 情感分析 (简单版)
        sentiment = self._analyze_sentiment(text)
        
        return ParsedInput(
            raw_text=text,
            entities=entities,
            relations=relations,
            timestamp=timestamp,
            location=location,
            main_event=main_event,
            sentiment=sentiment
        )
    
    def _extract_entities(self, text: str) -> List[ExtractedEntity]:
        """提取所有类型的实体"""
        entities = []
        
        # 人名识别
        person_patterns = [
            r'([\u4e00-\u9fa5]{2,4})(?:先生|女士|老师|医生|同学|朋友|同事)',
            r'([\u4e00-\u9fa5]{2,3})(?:和|与|跟|同)',
            r'(?:见到|遇到|碰到|约了)([\u4e00-\u9fa5]{2,4})',
        ]
        for pattern in person_patterns:
            for match in re.finditer(pattern, text):
                entities.append(ExtractedEntity(
                    text=match.group(1),
                    label="PERSON",
                    start=match.start(1),
                    end=match.end(1),
                    confidence=0.85
                ))
        
        # 地点识别
        place_patterns = [
            r'(星巴克|海底捞|肯德基|麦当劳|健身房|医院|公司|学校|咖啡厅|餐厅|商场|超市|公园|电影院)',
            r'在([\u4e00-\u9fa5]{2,6})(?:见面|吃饭|喝咖啡|锻炼|看病|上班|上课)',
            r'去([\u4e00-\u9fa5]{2,6})',
        ]
        for pattern in place_patterns:
            for match in re.finditer(pattern, text):
                place_name = match.group(1) if match.lastindex else match.group(0)
                entities.append(ExtractedEntity(
                    text=place_name,
                    label="PLACE",
                    start=match.start(),
                    end=match.end(),
                    confidence=0.90
                ))
        
        # 时间识别
        time_patterns = [
            (r'(\d{4}年\d{1,2}月\d{1,2}日)', 'DATE', 0.95),
            (r'(\d{1,2}月\d{1,2}日)', 'DATE', 0.90),
            (r'(今天|昨天|前天|明天|后天)', 'DATE', 0.95),
            (r'(早晨|上午|中午|下午|晚上|凌晨)(?:\d{1,2}点)?', 'TIME', 0.88),
            (r'(\d{1,2}点(?:\d{1,2}分)?)', 'TIME', 0.92),
            (r'(周一|周二|周三|周四|周五|周六|周日|星期[一二三四五六日])', 'DATE', 0.90),
        ]
        for pattern, label, conf in time_patterns:
            for match in re.finditer(pattern, text):
                entities.append(ExtractedEntity(
                    text=match.group(1),
                    label=label,
                    start=match.start(1),
                    end=match.end(1),
                    confidence=conf
                ))
        
        # 事件/活动识别
        event_keywords = [
            '开会', '会议', '讨论', '聊天', '吃饭', '聚餐', '锻炼', '健身', '看病',
            '约会', '见面', '上课', '培训', '演讲', '面试', '谈判', '签约', '庆祝'
        ]
        for keyword in event_keywords:
            for match in re.finditer(keyword, text):
                entities.append(ExtractedEntity(
                    text=keyword,
                    label="EVENT",
                    start=match.start(),
                    end=match.end(),
                    confidence=0.80
                ))
        
        # 物品/对象识别
        object_patterns = [
            r'([\u4e00-\u9fa5]{2,6})(?:咖啡|茶|饮料|酒|奶茶)',
            r'(拿铁|美式|卡布奇诺|可乐|雪碧|果汁)',
            r'(电脑|手机|笔记本|文件|资料|合同|报告)',
        ]
        for pattern in object_patterns:
            for match in re.finditer(pattern, text):
                obj_text = match.group(1) if match.lastindex else match.group(0)
                entities.append(ExtractedEntity(
                    text=obj_text,
                    label="OBJECT",
                    start=match.start(),
                    end=match.end(),
                    confidence=0.75
                ))
        
        return entities
    
    def _disambiguate_entities(self, entities: List[ExtractedEntity]) -> List[ExtractedEntity]:
        """实体消歧 - 将别名映射到标准形式"""
        for entity in entities:
            # 查找是否匹配已知别名
            for canonical, aliases in self.entity_aliases.items():
                if entity.text in aliases or any(self._similarity(entity.text, alias) > 0.8 for alias in aliases):
                    entity.normalized_form = canonical
                    entity.canonical_id = self._generate_concept_id(canonical, entity.label)
                    break
            else:
                # 新实体，生成ID
                entity.normalized_form = entity.text
                entity.canonical_id = self._generate_concept_id(entity.text, entity.label)
        
        return entities
    
    def _deduplicate_entities(self, entities: List[ExtractedEntity]) -> List[ExtractedEntity]:
        """实体去重 - 基于位置重叠和相似度"""
        if not entities:
            return entities
        
        # 按位置排序
        entities = sorted(entities, key=lambda e: (e.start, -e.confidence))
        
        deduplicated = []
        for entity in entities:
            # 检查是否与已保留的实体重叠
            is_duplicate = False
            for kept in deduplicated:
                # 位置重叠检查
                if not (entity.end <= kept.start or entity.start >= kept.end):
                    # 有重叠，保留置信度高的
                    if entity.confidence <= kept.confidence:
                        is_duplicate = True
                        break
                # 文本相似检查
                if self._similarity(entity.text, kept.text) > 0.9:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                deduplicated.append(entity)
        
        return deduplicated
    
    def _extract_relations(self, text: str, entities: List[ExtractedEntity]) -> List[ExtractedRelation]:
        """抽取实体间的关系"""
        relations = []
        
        # 找出PERSON和PLACE/EVENT的关系
        persons = [e for e in entities if e.label == "PERSON"]
        places = [e for e in entities if e.label == "PLACE"]
        events = [e for e in entities if e.label == "EVENT"]
        
        # 人在地点 (LOCATED_AT)
        for person in persons:
            for place in places:
                # 检查是否在同一个短句中
                if abs(person.start - place.start) < 20:
                    relations.append(ExtractedRelation(
                        subject=person.canonical_id or person.text,
                        predicate="LOCATED_AT",
                        object=place.canonical_id or place.text,
                        confidence=0.85
                    ))
        
        # 人参与活动 (PARTICIPATES_IN)
        for person in persons:
            for event in events:
                if abs(person.start - event.start) < 30:
                    relations.append(ExtractedRelation(
                        subject=person.canonical_id or person.text,
                        predicate="PARTICIPATES_IN",
                        object=event.text,
                        confidence=0.80
                    ))
        
        # 人和人的关系 (SOCIAL)
        if len(persons) >= 2:
            # 检查连接词
            social_keywords = ['和', '与', '跟', '同']
            for keyword in social_keywords:
                if keyword in text:
                    for i, p1 in enumerate(persons):
                        for p2 in persons[i+1:]:
                            relations.append(ExtractedRelation(
                                subject=p1.canonical_id or p1.text,
                                predicate="WITH",
                                object=p2.canonical_id or p2.text,
                                confidence=0.75
                            ))
        
        return relations
    
    def _parse_time(self, text: str, context: Optional[Dict]) -> Optional[datetime]:
        """解析时间"""
        now = context.get('current_time') if context else datetime.now()
        
        # 今天/昨天/明天
        if '今天' in text:
            return now
        elif '昨天' in text:
            return now.replace(day=now.day - 1)
        elif '明天' in text:
            return now.replace(day=now.day + 1)
        
        # 具体时间
        time_match = re.search(r'(\d{1,2})点(?:\d{1,2}分)?', text)
        if time_match:
            hour = int(time_match.group(1))
            return now.replace(hour=hour, minute=0, second=0)
        
        return now
    
    def _parse_location(self, text: str, entities: List[ExtractedEntity]) -> Optional[Dict]:
        """解析地点"""
        places = [e for e in entities if e.label == "PLACE"]
        if places:
            # 取第一个地点
            place = places[0]
            # 这里应该查询地点数据库获取坐标
            return {
                "name": place.normalized_form or place.text,
                "lat": 39.9,  # 默认值
                "lng": 116.4
            }
        return None
    
    def _identify_main_event(self, text: str, entities: List[ExtractedEntity]) -> Optional[str]:
        """识别主要事件"""
        events = [e for e in entities if e.label == "EVENT"]
        if events:
            # 返回最具体的（最长的）事件
            return max(events, key=lambda e: len(e.text)).text
        return None
    
    def _analyze_sentiment(self, text: str) -> Optional[str]:
        """简单情感分析"""
        positive_words = ['开心', '高兴', '愉快', '棒', '好', '喜欢', '爱', '成功']
        negative_words = ['难过', '伤心', '失败', '糟糕', '坏', '讨厌', '累', '烦']
        
        pos_count = sum(1 for w in positive_words if w in text)
        neg_count = sum(1 for w in negative_words if w in text)
        
        if pos_count > neg_count:
            return "positive"
        elif neg_count > pos_count:
            return "negative"
        return "neutral"
    
    def _similarity(self, a: str, b: str) -> float:
        """计算字符串相似度"""
        return SequenceMatcher(None, a, b).ratio()
    
    def _generate_concept_id(self, text: str, label: str) -> str:
        """生成概念ID"""
        hash_input = f"{label}:{text}"
        return f"{label.lower()}_{hashlib.md5(hash_input.encode()).hexdigest()[:8]}"


# 测试
if __name__ == "__main__":
    parser = UnifiedInputParser()
    
    test_inputs = [
        "今天早晨8点在星巴克和中伟讨论项目方案",
        "下午和李四去医院看牙医，预约了王医生",
        "明天晚上在海底捞和女朋友庆祝生日",
        "昨天在健身房遇到张伟，一起锻炼了1小时",
    ]
    
    for text in test_inputs:
        print("=" * 60)
        print(f"输入: {text}")
        print("-" * 60)
        
        result = parser.parse(text)
        
        print(f"\n实体 ({len(result.entities)}个):")
        for e in result.entities:
            norm = f" -> {e.normalized_form}" if e.normalized_form != e.text else ""
            print(f"  [{e.label}] {e.text}{norm} (置信度: {e.confidence:.2f})")
        
        print(f"\n关系 ({len(result.relations)}个):")
        for r in result.relations:
            print(f"  {r.subject} --{r.predicate}--> {r.object}")
        
        print(f"\n时间: {result.timestamp}")
        print(f"地点: {result.location}")
        print(f"主要事件: {result.main_event}")
        print(f"情感: {result.sentiment}")
