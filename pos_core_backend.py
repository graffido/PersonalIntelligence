#!/usr/bin/env python3
"""
POS Core v2.1 - Python实现 (增强版)
功能与C++后端等效，可作为原型使用

改进:
- 混合策略实体提取 (规则 + ML Service + LLM)
- 增强时间/地点识别
- 自动合并重叠实体
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass
from datetime import datetime, timedelta
import hashlib
import json
import os
import re
import requests
from enum import Enum

app = FastAPI(title="POS Core", version="2.1.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 数据存储 (内存中，重启丢失，实际应用应使用数据库)
DATA_DIR = "/tmp/pos_core_data"
os.makedirs(DATA_DIR, exist_ok=True)

memory_db: Dict[str, Dict] = {}
concept_db: Dict[str, Dict] = {}
relation_db: List[Dict] = {}

# ML Service 配置
ML_SERVICE_URL = os.getenv("ML_SERVICE_URL", "http://localhost:8000")
USE_ML_NER = os.getenv("USE_ML_NER", "true").lower() == "true"

# ============ 模型定义 ============

class UnifiedInputRequest(BaseModel):
    text: str
    time: Optional[str] = None
    location: Optional[Dict] = None
    source: Optional[str] = "user_input"

class QueryRequest(BaseModel):
    text: str
    type: str = "auto"  # auto, semantic, temporal, spatial, concept

class RecommendationRequest(BaseModel):
    context: Optional[Dict] = None
    limit: int = 5

class EntityExtractionStrategy(str, Enum):
    FAST = "fast"       # 仅本地规则
    HYBRID = "hybrid"   # 混合策略 (默认)
    ML = "ml"           # 仅 ML Service
    LLM = "llm"         # 仅 LLM

@dataclass
class Entity:
    text: str
    label: str
    start: int
    end: int
    confidence: float
    normalized: str
    source: str = "rule"  # rule, ml, llm

# ============ 增强实体提取系统 ============

class EntityExtractor:
    """混合策略实体提取器"""
    
    # 中文姓氏 (用于识别人名)
    CHINESE_SURNAMES = set(
        "王李张刘陈杨黄赵周吴徐孙马朱胡郭何林罗高郑梁谢宋唐许韩冯邓曹彭曾肖田董袁潘于蒋蔡余杜叶程苏魏吕丁任"
        "沈姚卢姜崔钟谭陆汪范金石廖贾夏韦傅方白邹孟熊秦邱江尹薛闫段雷侯龙史黎贺顾毛郝龚邵万钱"
    )
    
    # 常见中文名字后缀 (用于识别可能的人名)
    NAME_SUFFIXES = ["先生", "女士", "小姐", "老师", "医生", "经理", "总", "董"]
    
    # 关系称谓 (家庭成员、伴侣等)
    RELATION_TITLES = {
        "老婆": 0.95, "老公": 0.95, "妻子": 0.95, "丈夫": 0.95,
        "爸爸": 0.95, "妈妈": 0.95, "父亲": 0.95, "母亲": 0.95,
        "爸妈": 0.90, "父母": 0.90,
        "哥哥": 0.95, "姐姐": 0.95, "弟弟": 0.95, "妹妹": 0.95,
        "儿子": 0.95, "女儿": 0.95, "孩子": 0.85,
        "爷爷": 0.95, "奶奶": 0.95, "外公": 0.95, "外婆": 0.95,
        "岳父": 0.95, "岳母": 0.95, "公公": 0.95, "婆婆": 0.95,
        "男朋友": 0.95, "女朋友": 0.95, "男友": 0.95, "女友": 0.95,
        "同事": 0.85, "朋友": 0.80, "同学": 0.80,
    }
    
    # 地点后缀
    PLACE_SUFFIXES = [
        "店", "餐厅", "咖啡馆", "咖啡厅", "酒店", "机场", "车站", "医院", "学校",
        "公园", "广场", "大厦", "中心", "大楼", "小区", "公寓", "别墅", "馆",
        "厅", "室", "园", "场", "院", "所", "部", "处", "局", "公司", "银行"
    ]
    
    # 时间关键词
    TIME_KEYWORDS = [
        "早上", "上午", "中午", "下午", "晚上", "凌晨", "清晨", "傍晚", "夜间",
        "早晨", "午后", "黄昏", "半夜", "正午", "子夜"
    ]
    
    DATE_KEYWORDS = [
        "今天", "昨天", "明天", "后天", "前天", "大后天", "周末", "工作日",
        "周一", "周二", "周三", "周四", "周五", "周六", "周日", "星期",
        "这周一", "这周二", "这周三", "这周四", "这周五", "下周", "上周"
    ]

    def __init__(self):
        self.session = requests.Session()
        self.session.timeout = 5
        
    def extract(self, text: str, strategy: EntityExtractionStrategy = EntityExtractionStrategy.HYBRID) -> List[Entity]:
        """
        根据策略提取实体
        
        Args:
            text: 输入文本
            strategy: 提取策略
            
        Returns:
            实体列表
        """
        if strategy == EntityExtractionStrategy.FAST:
            return self._extract_fast(text)
        
        elif strategy == EntityExtractionStrategy.ML:
            return self._extract_ml(text) or self._extract_fast(text)
        
        elif strategy == EntityExtractionStrategy.LLM:
            return self._extract_llm(text) or self._extract_fast(text)
        
        else:  # HYBRID
            return self._extract_hybrid(text)
    
    def _extract_fast(self, text: str) -> List[Entity]:
        """
        快速本地提取 - 增强版正则 + 词典
        """
        entities = []
        
        # 1. 关系称谓 (如"老婆"、"爸爸")
        entities.extend(self._extract_relation_titles(text))
        
        # 2. 时间实体 (高置信度)
        entities.extend(self._extract_time_patterns(text))
        
        # 3. 联系信息 (正则，高置信度)
        entities.extend(self._extract_contact_patterns(text))
        
        # 4. 人名识别 (基于姓氏 + 上下文)
        entities.extend(self._extract_person_names(text))
        
        # 5. 地点识别 (基于后缀)
        entities.extend(self._extract_places(text))
        
        # 6. 组织和公司
        entities.extend(self._extract_organizations(text))
        
        # 7. 事件/活动
        entities.extend(self._extract_events(text))
        
        # 去重和排序
        return self._deduplicate_entities(entities)

    def _extract_relation_titles(self, text: str) -> List[Entity]:
        """提取关系称谓 (老婆、爸爸、同事等)"""
        entities = []
        matched_positions = set()  # 记录已匹配的位置，避免重叠
        
        # 按长度排序，先匹配长的（避免"男朋友"被拆成"男"+"朋友"）
        sorted_titles = sorted(self.RELATION_TITLES.items(), key=lambda x: -len(x[0]))
        
        for title, conf in sorted_titles:
            idx = 0
            while True:
                idx = text.find(title, idx)
                if idx == -1:
                    break
                
                # 检查这个位置是否已经被其他更长的匹配占用
                position_range = set(range(idx, idx + len(title)))
                if position_range & matched_positions:  # 有重叠
                    idx += 1
                    continue
                
                # 匹配成功
                entities.append(Entity(
                    text=title,
                    label='PERSON',
                    start=idx,
                    end=idx + len(title),
                    confidence=conf,
                    normalized=title,
                    source="rule"
                ))
                matched_positions.update(position_range)
                idx += len(title)
        
        return entities
    
    def _extract_time_patterns(self, text: str) -> List[Entity]:
        """提取时间相关实体"""
        entities = []
        
        # 日期模式
        date_patterns = [
            # 标准日期: 2024年1月15日, 2024-01-15, 2024/01/15
            (r'(\d{4}年\d{1,2}月\d{1,2}[日号]?)', 'DATE', 0.95),
            (r'(\d{4}[-/]\d{1,2}[-/]\d{1,2})', 'DATE', 0.95),
            # 月份日期: 1月15日
            (r'(\d{1,2}月\d{1,2}[日号])', 'DATE', 0.90),
            # 相对日期
            (r'(今天|昨天|明天|后天|大后天|前天)', 'DATE', 0.95),
            # 周几
            (r'(星期[一二三四五六日天]|周[一二三四五六日])', 'DATE', 0.92),
            (r'(上周|这周|本周|下周)[一二三四五六日]?', 'DATE', 0.90),
            # 时间段
            (r'(早(?:晨|上)|上午|中午|下午|晚上|凌晨|傍晚|夜间)', 'TIME', 0.88),
            # 具体时间: 14:30, 下午3点, 3点半
            (r'(\d{1,2}[:：]\d{2})', 'TIME', 0.95),
            (r'([上下]午\s*\d{1,2}\s*[点：](?:\d{1,2}[分]?)?)', 'TIME', 0.90),
            (r'(\d{1,2}\s*点\s*(?:半|\d{1,2}\s*分)?)', 'TIME', 0.88),
        ]
        
        for pattern, label, conf in date_patterns:
            for match in re.finditer(pattern, text):
                entities.append(Entity(
                    text=match.group(1),
                    label=label,
                    start=match.start(1),
                    end=match.end(1),
                    confidence=conf,
                    normalized=self._normalize_time(match.group(1)),
                    source="rule"
                ))
        
        return entities
    
    def _extract_contact_patterns(self, text: str) -> List[Entity]:
        """提取联系信息"""
        entities = []
        
        contact_patterns = [
            # 邮箱
            (r'([\w\.-]+@[\w\.-]+\.\w+)', 'EMAIL', 0.98),
            # 中国手机号
            (r'(1[3-9]\d{9})', 'PHONE', 0.98),
            # 座机
            (r'(0\d{2,3}[-]?\d{7,8})', 'PHONE', 0.95),
            # URL
            (r'(https?://[^\s\u3002\uff0c]+)', 'URL', 0.97),
            # IP地址
            (r'(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})', 'IP', 0.95),
        ]
        
        for pattern, label, conf in contact_patterns:
            for match in re.finditer(pattern, text):
                entities.append(Entity(
                    text=match.group(1),
                    label=label,
                    start=match.start(1),
                    end=match.end(1),
                    confidence=conf,
                    normalized=match.group(1),
                    source="rule"
                ))
        
        return entities
    
    def _extract_person_names(self, text: str) -> List[Entity]:
        """识别人名"""
        entities = []
        
        # 时间相关字符，不应出现在人名中
        time_chars = set('天晚早午晨夜点分秒周月日年')
        # 常见介词/动词，人名后不应直接跟这些
        stop_chars = set('的在是和与跟到去就来有要约')
        
        # 策略1: 姓氏 + 1-2个汉字 (可能是人名)
        for surname in self.CHINESE_SURNAMES:
            # 匹配姓氏后跟1-2个字符
            pattern = surname + '[\\u4e00-\\u9fa5]{1,2}'
            
            for match in re.finditer(pattern, text):
                name_start = match.start()
                name_end = match.end()
                full_name = text[name_start:name_end]
                
                # 过滤常见词
                if self._is_common_word(full_name):
                    continue
                
                # ===== 关键修复1: 避免时间词汇 =====
                # 检查名字部分是否包含时间字符
                name_body = full_name[1:]  # 去掉姓氏
                has_time_char = any(c in time_chars for c in name_body)
                if has_time_char:
                    continue  # 跳过包含时间字符的"名字"(如"周六晚"中的"晚")
                
                # ===== 关键修复2: 处理介词后缀 =====
                # 检查最后一个字是否是介词，如果是则去掉
                original_end = name_end
                while len(full_name) > 1 and full_name[-1] in stop_chars:
                    full_name = full_name[:-1]
                    name_end -= 1
                
                # 如果去掉介词后只剩姓氏，跳过
                if len(full_name) < 2:
                    continue
                
                # 检查下一个字符是否是介词（确认是否切分正确）
                if name_end < len(text) and text[name_end] in stop_chars:
                    pass  # 正确切分，下一个字符是介词
                
                # 检查上下文 (后面跟着称谓更可能是人名)
                context_after = text[name_end:name_end+4]
                has_title = any(s in context_after for s in self.NAME_SUFFIXES)
                conf = 0.85 if has_title else 0.70
                
                # 避免重复添加
                already_exists = any(e.start == name_start and e.end == name_end for e in entities)
                if already_exists:
                    continue
                
                entities.append(Entity(
                    text=full_name,
                    label='PERSON',
                    start=name_start,
                    end=name_end,
                    confidence=conf,
                    normalized=full_name,
                    source="rule"
                ))
        
        # 策略2: 称谓模式 (X先生, X女士) - 更严格的边界
        title_pattern = r'([\\u4e00-\\u9fa5]{1,2})(先生|女士|小姐|老师|医生|教授)(?![\\u4e00-\\u9fa5])'
        for match in re.finditer(title_pattern, text):
            full = match.group(0)
            name = match.group(1)
            entities.append(Entity(
                text=full,
                label='PERSON',
                start=match.start(),
                end=match.end(),
                confidence=0.90,
                normalized=name,
                source="rule"
            ))
        
        # 策略3: 常见英文名字
        english_names = r'\b([A-Z][a-z]{1,10})\b'
        for match in re.finditer(english_names, text):
            name = match.group(1)
            common_words = {"The", "This", "That", "There", "Today", "Tomorrow", "Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"}
            if name not in common_words:
                entities.append(Entity(
                    text=name,
                    label='PERSON',
                    start=match.start(),
                    end=match.end(),
                    confidence=0.65,
                    normalized=name,
                    source="rule"
                ))
        
        return entities
    
    def _extract_places(self, text: str) -> List[Entity]:
        """识别地点"""
        entities = []
        
        # 策略1: 基于后缀的地点识别
        # 匹配 X店, X餐厅, XX咖啡厅 等
        place_pattern = r'([\\u4e00-\\u9fa5]{2,8}(?:' + '|'.join(self.PLACE_SUFFIXES) + r'))'
        for match in re.finditer(place_pattern, text):
            place = match.group(1)
            entities.append(Entity(
                text=place,
                label='PLACE',
                start=match.start(),
                end=match.end(),
                confidence=0.82,
                normalized=place,
                source="rule"
            ))
        
        # 策略2: 特定地点类型
        specific_places = [
            (r'(家|家里|家中)', 'PLACE', 0.90),
            (r'(公司|单位|办公室)', 'PLACE', 0.90),
            (r'(学校|大学|学院|中学|小学)', 'PLACE', 0.88),
            (r'(北京|上海|广州|深圳|杭州|南京|成都|武汉|西安|重庆|天津|苏州)', 'GPE', 0.95),
            (r'(中国|美国|日本|韩国|英国|法国|德国|意大利|西班牙|加拿大|澳大利亚)', 'GPE', 0.95),
        ]
        for pattern, label, conf in specific_places:
            for match in re.finditer(pattern, text):
                entities.append(Entity(
                    text=match.group(1),
                    label=label,
                    start=match.start(1),
                    end=match.end(1),
                    confidence=conf,
                    normalized=match.group(1),
                    source="rule"
                ))
        
        return entities
    
    def _extract_organizations(self, text: str) -> List[Entity]:
        """识别组织和公司"""
        entities = []
        
        org_patterns = [
            # 公司类型
            (r'([\\u4e00-\\u9fa5]{2,10}(?:科技|技术|网络|软件|信息|咨询|文化|传媒|教育|医疗|金融|投资|集团|股份|有限)公司?)', 'ORG', 0.85),
            (r'([\\u4e00-\\u9fa5]{2,10}(?:集团|银行|证券|保险|基金|信托))', 'ORG', 0.88),
            # 互联网产品/平台
            (r'(微信|微博|抖音|小红书|淘宝|天猫|京东|拼多多|美团|滴滴|支付宝|网易云音乐|QQ|B站|知乎)', 'PRODUCT', 0.95),
            # 品牌
            (r'(苹果|华为|小米|三星|索尼|耐克|阿迪达斯|星巴克|麦当劳|肯德基)', 'BRAND', 0.90),
        ]
        
        for pattern, label, conf in org_patterns:
            for match in re.finditer(pattern, text):
                entities.append(Entity(
                    text=match.group(1),
                    label=label,
                    start=match.start(1),
                    end=match.end(1),
                    confidence=conf,
                    normalized=match.group(1),
                    source="rule"
                ))
        
        return entities
    
    def _extract_events(self, text: str) -> List[Entity]:
        """识别事件/活动"""
        entities = []
        
        event_patterns = [
            (r'(会议|例会|周会|月会|年会|晨会|夕会|讨论会|评审会|发布会)', 'EVENT', 0.88),
            (r'(聚会|聚餐|约会|派对|晚宴|午餐|晚餐|早餐|下午茶)', 'EVENT', 0.85),
            (r'(面试|考试|测试|考核|评估|答辩)', 'EVENT', 0.87),
            (r'(旅行|旅游|出差|度假|郊游|露营|登山|跑步|健身|游泳)', 'EVENT', 0.82),
            (r'(电影|演唱会|展览|演出|比赛|球赛|音乐会)', 'EVENT', 0.85),
            (r'(生日|纪念日|节日|假期|周末|周年庆)', 'EVENT', 0.88),
        ]
        
        for pattern, label, conf in event_patterns:
            for match in re.finditer(pattern, text):
                entities.append(Entity(
                    text=match.group(1),
                    label=label,
                    start=match.start(1),
                    end=match.end(1),
                    confidence=conf,
                    normalized=match.group(1),
                    source="rule"
                ))
        
        return entities
    
    def _extract_ml(self, text: str) -> Optional[List[Entity]]:
        """调用 ML Service 进行实体提取"""
        if not USE_ML_NER:
            return None
        
        try:
            response = self.session.post(
                f"{ML_SERVICE_URL}/ner/extract",
                json={"text": text, "extract_relations": False},
                timeout=3
            )
            
            if response.status_code == 200:
                result = response.json()
                entities = []
                for e in result.get("entities", []):
                    entities.append(Entity(
                        text=e["text"],
                        label=e["label"],
                        start=e.get("start", 0),
                        end=e.get("end", 0),
                        confidence=e.get("confidence", 0.8),
                        normalized=e.get("normalized", e["text"]),
                        source="ml"
                    ))
                return entities
                
        except Exception as e:
            print(f"ML NER error: {e}")
        
        return None
    
    def _extract_llm(self, text: str) -> Optional[List[Entity]]:
        """使用 LLM 提取实体"""
        # 如果文本很长，避免使用 LLM (成本高)
        if len(text) > 200:
            return None
        
        prompt = f"""从以下文本中提取实体，返回 JSON 数组格式。
只返回 JSON，不要有其他内容。

文本: {text}

实体类型:
- PERSON: 人名
- PLACE: 地点
- ORG: 组织/公司
- DATE: 日期
- TIME: 时间
- EVENT: 事件/活动
- PRODUCT: 产品

返回格式示例:
[{{"text": "张三", "label": "PERSON", "start": 0, "end": 2}}]

提取结果:"""

        try:
            response = self.session.post(
                f"{ML_SERVICE_URL}/llm/chat",
                json={
                    "messages": prompt,
                    "temperature": 0.1,
                    "max_tokens": 500
                },
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result.get("content", "[]")
                
                # 解析 JSON
                import json
                try:
                    raw_entities = json.loads(content)
                    entities = []
                    for e in raw_entities:
                        entities.append(Entity(
                            text=e["text"],
                            label=e["label"],
                            start=e.get("start", text.find(e["text"])),
                            end=e.get("end", text.find(e["text"]) + len(e["text"])),
                            confidence=0.85,
                            normalized=e["text"],
                            source="llm"
                        ))
                    return entities
                except json.JSONDecodeError:
                    pass
                    
        except Exception as e:
            print(f"LLM NER error: {e}")
        
        return None
    
    def _extract_hybrid(self, text: str) -> List[Entity]:
        """混合策略：本地 + ML Service"""
        # 1. 先进行本地快速提取
        fast_entities = self._extract_fast(text)
        
        # 2. 如果实体数量较少或文本较长，尝试 ML Service
        ml_entities = None
        if len(fast_entities) < 2 or len(text) > 30:
            ml_entities = self._extract_ml(text)
        
        # 3. 合并结果
        if ml_entities:
            return self._merge_entities(fast_entities, ml_entities)
        
        return fast_entities
    
    def _merge_entities(self, local: List[Entity], ml: List[Entity]) -> List[Entity]:
        """合并本地和 ML 提取的实体，去除重叠"""
        all_entities = local + ml
        
        # 按位置排序，置信度高的优先
        all_entities.sort(key=lambda e: (e.start, -e.confidence))
        
        # 去除重叠
        result = []
        used_ranges: List[Tuple[int, int]] = []
        
        for entity in all_entities:
            # 检查是否与已选实体重叠
            overlap = False
            for start, end in used_ranges:
                if not (entity.end <= start or entity.start >= end):
                    # 有重叠，但如果是不同类型可以保留
                    if entity.label == 'PERSON' or entity.label == 'PLACE':
                        continue  # 跳过重叠的低优先级实体
                    overlap = True
                    break
            
            if not overlap:
                result.append(entity)
                used_ranges.append((entity.start, entity.end))
        
        # 按原始位置重新排序
        result.sort(key=lambda e: e.start)
        return result
    
    def _deduplicate_entities(self, entities: List[Entity]) -> List[Entity]:
        """去重并排序"""
        # 按位置排序
        entities.sort(key=lambda e: e.start)
        
        # 去重 (相同位置只保留置信度最高的)
        seen = {}  # (start, end) -> Entity
        for e in entities:
            key = (e.start, e.end)
            if key not in seen or seen[key].confidence < e.confidence:
                seen[key] = e
        
        return list(seen.values())
    
    def _is_common_word(self, text: str) -> bool:
        """检查是否是常用词 (避免误判)"""
        common_words = {
            "我们", "你们", "他们", "她们", "它们", "人们",
            "这里", "那里", "哪里", "非常", "很多", "不少",
            "一些", "一起", "一定", "一样", "一般", "一直",
            "可以", "可能", "可是", "可惜", "可怕",
            "就是", "都有", "都有", "还有", "没有",
            "如果", "因为", "所以", "虽然", "但是",
            "自己", "别人", "大家", "各位", "各位",
            "什么", "怎么", "多少", "几", "谁", "哪"
        }
        return text in common_words
    
    def _normalize_time(self, text: str) -> str:
        """标准化时间表达"""
        # 转换为标准格式，例如将 "明天" 转换为实际日期
        if text in ["今天"]:
            return datetime.now().strftime("%Y-%m-%d")
        elif text in ["昨天"]:
            return (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        elif text in ["明天"]:
            return (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
        elif text in ["后天"]:
            return (datetime.now() + timedelta(days=2)).strftime("%Y-%m-%d")
        return text


# 全局实体提取器实例
extractor = EntityExtractor()

# ============ 兼容原有接口 ============

def extract_entities(text: str, strategy: str = "hybrid") -> List[Dict]:
    """
    实体提取主函数 (兼容原有接口)
    
    Args:
        text: 输入文本
        strategy: 提取策略 (fast, hybrid, ml, llm)
    
    Returns:
        实体字典列表
    """
    try:
        strategy_enum = EntityExtractionStrategy(strategy)
    except ValueError:
        strategy_enum = EntityExtractionStrategy.HYBRID
    
    entities = extractor.extract(text, strategy_enum)
    
    # 转换为原有格式
    return [
        {
            "text": e.text,
            "label": e.label,
            "start": e.start,
            "end": e.end,
            "confidence": e.confidence,
            "normalized": e.normalized,
            "source": e.source
        }
        for e in entities
    ]

# ============ 其余原有代码保持不变 ============

def get_or_create_concept(label: str, concept_type: str) -> str:
    """获取或创建概念"""
    concept_id = f"{concept_type.lower()}_{hashlib.md5(label.encode()).hexdigest()[:8]}"
    
    if concept_id not in concept_db:
        concept_db[concept_id] = {
            "id": concept_id,
            "type": concept_type,
            "label": label,
            "aliases": [label],
            "memories": [],
            "relations": [],
            "created_at": datetime.now().isoformat()
        }
    
    return concept_id

def apply_reasoning(memory: Dict) -> List[Dict]:
    """应用推理规则"""
    results = []
    
    # 规则1: 时间冲突检测
    for mem_id, mem in memory_db.items():
        if mem["id"] != memory["id"]:
            if memory.get("timestamp") and mem.get("timestamp"):
                time_diff = abs(
                    datetime.fromisoformat(memory["timestamp"]) - 
                    datetime.fromisoformat(mem["timestamp"])
                ).total_seconds() / 3600
                
                if time_diff < 1:
                    results.append({
                        "rule_id": "schedule_conflict",
                        "rule_name": "日程冲突检测",
                        "type": "conflict",
                        "description": f"与记忆 '{mem['content'][:20]}...' 时间接近",
                        "confidence": 0.85,
                        "severity": "medium"
                    })
    
    # 规则2: 模式识别
    concepts = memory.get("concepts", [])
    for concept_id in concepts:
        if concept_id in concept_db:
            concept = concept_db[concept_id]
            if len(concept["memories"]) >= 3:
                results.append({
                    "rule_id": "habit_pattern",
                    "rule_name": "习惯模式",
                    "type": "pattern",
                    "description": f"经常涉及 '{concept['label']}'，已记录 {len(concept['memories'])} 次",
                    "confidence": min(len(concept["memories"]) / 10, 0.9)
                })
    
    return results

def generate_recommendations(context: Optional[Dict]) -> List[Dict]:
    """生成推荐"""
    recommendations = []
    
    now = datetime.now()
    hour = now.hour
    
    if 6 <= hour < 9:
        recommendations.append({
            "type": "time_based",
            "title": "早晨习惯",
            "description": "您经常在早晨处理重要事务",
            "confidence": 0.75,
            "priority": 3,
            "reason": "基于历史时间模式"
        })
    
    upcoming = [
        m for m in memory_db.values()
        if m.get("timestamp") and 
        abs(datetime.fromisoformat(m["timestamp"]) - now).total_seconds() < 86400
    ]
    
    if len(upcoming) >= 3:
        recommendations.append({
            "type": "conflict_warning",
            "title": "日程提醒",
            "description": f"未来24小时有 {len(upcoming)} 个安排，注意时间冲突",
            "confidence": 0.9,
            "priority": 5,
            "reason": "基于日程密度"
        })
    
    for concept_id, concept in concept_db.items():
        if concept["type"] == "PERSON" and len(concept["memories"]) >= 2:
            recommendations.append({
                "type": "social_based",
                "title": "社交提醒",
                "description": f"已经一段时间没有和 '{concept['label']}' 联系了",
                "confidence": 0.6,
                "priority": 2,
                "reason": "基于社交频率"
            })
            break
    
    return recommendations

# ============ API端点 ============

@app.get("/health")
def health():
    """健康检查"""
    # 检查 ML Service 是否可用
    ml_available = False
    try:
        response = requests.get(f"{ML_SERVICE_URL}/health", timeout=2)
        ml_available = response.status_code == 200
    except:
        pass
    
    return {
        "status": "ok",
        "version": "2.1.0",
        "core": "pos_core_python",
        "features": [
            "unified_input",
            "entity_extraction",
            "reasoning",
            "recommendation",
            "prediction"
        ],
        "entity_extraction": {
            "mode": "hybrid",
            "ml_service_available": ml_available,
            "strategies": ["fast", "hybrid", "ml", "llm"]
        },
        "stats": {
            "memories": len(memory_db),
            "concepts": len(concept_db),
            "relations": len(relation_db)
        }
    }

@app.post("/api/v1/input")
def process_input(req: UnifiedInputRequest):
    """统一输入处理"""
    # 1. 提取实体 (使用混合策略)
    entities = extract_entities(req.text, "hybrid")
    
    # 2. 创建/获取概念
    concept_ids = []
    for entity in entities:
        cid = get_or_create_concept(entity["normalized"], entity["label"])
        concept_ids.append(cid)
        entity["concept_id"] = cid
        concept_db[cid]["memories"].append(f"mem_{len(memory_db)}")
    
    # 3. 创建记忆
    memory_id = f"mem_{len(memory_db)}"
    memory = {
        "id": memory_id,
        "content": req.text,
        "timestamp": req.time or datetime.now().isoformat(),
        "location": req.location,
        "entities": entities,
        "concepts": concept_ids,
        "source": req.source,
        "created_at": datetime.now().isoformat()
    }
    memory_db[memory_id] = memory
    
    # 4. 执行推理
    reasoning_results = apply_reasoning(memory)
    
    # 5. 生成推荐
    ctx = {"current_time": datetime.now().isoformat()}
    recommendations = generate_recommendations(ctx)
    
    return {
        "success": True,
        "memory_id": memory_id,
        "entities": entities,
        "entity_count": len(entities),
        "reasoning_results": reasoning_results,
        "recommendations": recommendations,
        "stats": {
            "new_concepts": len([c for c in concept_ids if len(concept_db[c]["memories"]) == 1]),
            "linked_concepts": len([c for c in concept_ids if len(concept_db[c]["memories"]) > 1])
        }
    }

@app.post("/api/v1/query")
def query(req: QueryRequest):
    """智能查询"""
    entities = extract_entities(req.text)
    
    strategy = req.type
    if strategy == "auto":
        has_person = any(e["label"] == "PERSON" for e in entities)
        strategy = "concept" if has_person else "semantic"
    
    results = []
    
    if strategy == "concept":
        for entity in entities:
            if entity["label"] in ["PERSON", "PLACE", "ORG"]:
                cid = get_or_create_concept(entity["normalized"], entity["label"])
                for mem_id in concept_db[cid]["memories"]:
                    if mem_id in memory_db:
                        results.append({
                            "memory_id": mem_id,
                            "content": memory_db[mem_id]["content"],
                            "match_type": "concept",
                            "concept": entity["normalized"]
                        })
    else:
        for mem_id, mem in memory_db.items():
            if any(e["normalized"] in req.text for e in mem["entities"]):
                results.append({
                    "memory_id": mem_id,
                    "content": mem["content"],
                    "match_type": "semantic"
                })
    
    seen = set()
    unique_results = []
    for r in results:
        if r["memory_id"] not in seen:
            seen.add(r["memory_id"])
            unique_results.append(r)
    
    return {
        "strategy": strategy,
        "entities": [{"text": e["text"], "label": e["label"], "source": e.get("source", "rule")} for e in entities],
        "results": unique_results[:10],
        "count": len(unique_results)
    }

@app.post("/api/v1/recommendations")
def get_recommendations(req: RecommendationRequest):
    """获取推荐"""
    recommendations = generate_recommendations(req.context)
    return {
        "recommendations": recommendations[:req.limit],
        "count": len(recommendations)
    }

@app.get("/api/v1/predictions")
def get_predictions(hours: int = 24):
    """获取预测"""
    predictions = []
    
    for concept_id, concept in concept_db.items():
        if len(concept["memories"]) >= 3:
            predictions.append({
                "event_type": f"与 {concept['label']} 相关的活动",
                "confidence": min(len(concept["memories"]) / 10, 0.9),
                "description": f"基于历史记录，您经常与 '{concept['label']}' 互动"
            })
    
    return {"predictions": predictions[:5]}

@app.get("/api/v1/stats")
def get_stats():
    """获取统计"""
    entity_sources = {}
    for mem in memory_db.values():
        for e in mem.get("entities", []):
            source = e.get("source", "rule")
            entity_sources[source] = entity_sources.get(source, 0) + 1
    
    return {
        "memories": len(memory_db),
        "concepts": len(concept_db),
        "relations": len(relation_db),
        "entity_sources": entity_sources,
        "concept_types": {
            "PERSON": len([c for c in concept_db.values() if c["type"] == "PERSON"]),
            "PLACE": len([c for c in concept_db.values() if c["type"] == "PLACE"]),
            "ORG": len([c for c in concept_db.values() if c["type"] == "ORG"]),
            "EVENT": len([c for c in concept_db.values() if c["type"] == "EVENT"])
        }
    }

@app.get("/api/v1/concepts/{concept_id}")
def get_concept(concept_id: str):
    """获取概念详情"""
    if concept_id not in concept_db:
        return {"error": "Concept not found"}
    
    concept = concept_db[concept_id]
    return {
        "id": concept["id"],
        "type": concept["type"],
        "label": concept["label"],
        "aliases": concept["aliases"],
        "memory_count": len(concept["memories"])
    }

@app.post("/api/v1/infer")
def trigger_inference():
    """手动触发推理"""
    all_results = []
    for mem in memory_db.values():
        results = apply_reasoning(mem)
        all_results.extend(results)
    
    return {
        "inference_results": all_results,
        "count": len(all_results)
    }

# 新增: 实体提取测试端点
@app.post("/api/v1/test/entities")
def test_entity_extraction(text: str, strategy: str = "hybrid"):
    """测试实体提取功能"""
    start_time = datetime.now()
    entities = extract_entities(text, strategy)
    elapsed = (datetime.now() - start_time).total_seconds() * 1000
    
    return {
        "text": text,
        "strategy": strategy,
        "entities": entities,
        "count": len(entities),
        "elapsed_ms": round(elapsed, 2)
    }

if __name__ == "__main__":
    import uvicorn
    
    print("=" * 60)
    print("POS Core v2.1 (Python) - 增强版实体提取")
    print("个人本体记忆系统 - 核心后端")
    print("=" * 60)
    print("\n改进功能:")
    print("  ✓ 混合策略实体提取 (规则 + ML + LLM)")
    print("  ✓ 智能人名识别 (姓氏 + 上下文)")
    print("  ✓ 增强时间/地点识别")
    print("  ✓ 自动合并重叠实体")
    print("  ✓ 实体来源追踪 (rule/ml/llm)")
    print("\n实体提取策略:")
    print("  - fast:   仅本地规则 (最快)")
    print("  - hybrid: 混合策略 (默认)")
    print("  - ml:     仅 ML Service")
    print("  - llm:    仅 LLM (最准但慢)")
    print("\nAPI端点:")
    print("  POST /api/v1/input           - 统一输入")
    print("  POST /api/v1/query           - 智能查询")
    print("  POST /api/v1/test/entities   - 实体提取测试")
    print("=" * 60)
    
    uvicorn.run(app, host="0.0.0.0", port=9000)
