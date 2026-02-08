"""
增强版NER服务 - 支持深度学习模型和Ontology建模
Enhanced NER Service with Deep Learning Models and Ontology Support
"""
import json
import re
import pickle
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from collections import defaultdict


class EntityType(Enum):
    """支持的实体类型"""
    PERSON = "PERSON"
    PLACE = "PLACE"
    EVENT = "EVENT"
    CONCEPT = "CONCEPT"
    ORGANIZATION = "ORGANIZATION"
    TIME = "TIME"
    MONEY = "MONEY"
    PRODUCT = "PRODUCT"
    WORK_OF_ART = "WORK_OF_ART"
    CUSTOM = "CUSTOM"


class RelationType(Enum):
    """支持的关系类型"""
    LOCATED_AT = "LOCATED_AT"  # 位于
    WORKS_FOR = "WORKS_FOR"    # 工作于
    PARTICIPATED_IN = "PARTICIPATED_IN"  # 参与
    CREATED_BY = "CREATED_BY"  # 由...创建
    RELATED_TO = "RELATED_TO"  # 相关
    PART_OF = "PART_OF"        # 属于
    HAS_PROPERTY = "HAS_PROPERTY"  # 具有属性
    CUSTOM = "CUSTOM"          # 自定义


@dataclass
class Entity:
    """实体类"""
    text: str
    type: EntityType
    start: int
    end: int
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: Optional[str] = None
    
    def __post_init__(self):
        if self.id is None:
            self.id = f"{self.type.value}_{hash(self.text) % 10000000:07d}"
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "text": self.text,
            "type": self.type.value,
            "start": self.start,
            "end": self.end,
            "confidence": self.confidence,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Entity':
        return cls(
            text=data["text"],
            type=EntityType(data["type"]),
            start=data["start"],
            end=data["end"],
            confidence=data.get("confidence", 1.0),
            metadata=data.get("metadata", {}),
            id=data.get("id")
        )


@dataclass
class Relation:
    """关系类"""
    subject: Entity
    predicate: RelationType
    object: Entity
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: Optional[str] = None
    
    def __post_init__(self):
        if self.id is None:
            self.id = f"REL_{hash(self.subject.text + self.predicate.value + self.object.text) % 10000000:07d}"
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "subject": self.subject.to_dict(),
            "predicate": self.predicate.value,
            "object": self.object.to_dict(),
            "confidence": self.confidence,
            "metadata": self.metadata
        }


@dataclass
class NERResult:
    """NER结果类"""
    text: str
    entities: List[Entity]
    relations: List[Relation] = field(default_factory=list)
    language: str = "unknown"
    
    def to_dict(self) -> Dict:
        return {
            "text": self.text,
            "language": self.language,
            "entities": [e.to_dict() for e in self.entities],
            "relations": [r.to_dict() for r in self.relations],
            "entity_count": len(self.entities),
            "relation_count": len(self.relations)
        }


class OntologyManager:
    """
    Ontology管理器 - 管理个人知识图谱
    支持增量学习和实体链接
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = Path(storage_path) if storage_path else Path("ontology_db")
        self.storage_path.mkdir(exist_ok=True)
        
        # 实体库: text -> Entity
        self.entity_library: Dict[str, Entity] = {}
        # 类型索引: type -> [Entity]
        self.type_index: Dict[EntityType, List[str]] = defaultdict(list)
        # 关系库: subject_id -> [Relation]
        self.relation_library: Dict[str, List[Relation]] = defaultdict(list)
        # 同义词映射
        self.synonyms: Dict[str, str] = {}
        # 自定义实体类型
        self.custom_types: Dict[str, Dict] = {}
        
        self._load_data()
    
    def _load_data(self):
        """加载持久化数据"""
        entity_file = self.storage_path / "entities.pkl"
        relation_file = self.storage_path / "relations.pkl"
        synonym_file = self.storage_path / "synonyms.json"
        custom_type_file = self.storage_path / "custom_types.json"
        
        if entity_file.exists():
            with open(entity_file, 'rb') as f:
                data = pickle.load(f)
                for e_data in data:
                    entity = Entity.from_dict(e_data)
                    self.entity_library[entity.text] = entity
                    self.type_index[entity.type].append(entity.text)
        
        if relation_file.exists():
            with open(relation_file, 'rb') as f:
                self.relation_library = pickle.load(f)
        
        if synonym_file.exists():
            with open(synonym_file, 'r', encoding='utf-8') as f:
                self.synonyms = json.load(f)
        
        if custom_type_file.exists():
            with open(custom_type_file, 'r', encoding='utf-8') as f:
                self.custom_types = json.load(f)
    
    def save_data(self):
        """保存数据到磁盘"""
        entity_data = [e.to_dict() for e in self.entity_library.values()]
        with open(self.storage_path / "entities.pkl", 'wb') as f:
            pickle.dump(entity_data, f)
        
        with open(self.storage_path / "relations.pkl", 'wb') as f:
            pickle.dump(self.relation_library, f)
        
        with open(self.storage_path / "synonyms.json", 'w', encoding='utf-8') as f:
            json.dump(self.synonyms, f, ensure_ascii=False, indent=2)
        
        with open(self.storage_path / "custom_types.json", 'w', encoding='utf-8') as f:
            json.dump(self.custom_types, f, ensure_ascii=False, indent=2)
    
    def add_entity(self, entity: Entity, merge: bool = True) -> Entity:
        """
        添加实体到Ontology
        
        Args:
            entity: 要添加的实体
            merge: 是否合并已存在的实体
        
        Returns:
            添加或合并后的实体
        """
        normalized_text = self._normalize_text(entity.text)
        
        # 检查同义词
        if normalized_text in self.synonyms:
            canonical_text = self.synonyms[normalized_text]
            if canonical_text in self.entity_library:
                existing = self.entity_library[canonical_text]
                if merge:
                    # 更新置信度
                    existing.confidence = max(existing.confidence, entity.confidence)
                    existing.metadata.update(entity.metadata)
                    return existing
                return existing
        
        # 检查是否已存在
        if normalized_text in self.entity_library:
            existing = self.entity_library[normalized_text]
            if merge:
                existing.confidence = max(existing.confidence, entity.confidence)
                existing.metadata.update(entity.metadata)
                return existing
            return existing
        
        # 添加新实体
        self.entity_library[normalized_text] = entity
        self.type_index[entity.type].append(normalized_text)
        return entity
    
    def add_synonym(self, synonym: str, canonical: str):
        """添加同义词映射"""
        self.synonyms[self._normalize_text(synonym)] = self._normalize_text(canonical)
    
    def add_custom_type(self, type_name: str, description: str, examples: List[str] = None):
        """添加自定义实体类型"""
        self.custom_types[type_name] = {
            "description": description,
            "examples": examples or [],
            "created_at": str(np.datetime64('now'))
        }
    
    def link_entity(self, text: str) -> Optional[Entity]:
        """
        实体链接 - 将文本链接到Ontology中的实体
        
        Args:
            text: 要链接的文本
        
        Returns:
            链接到的实体，如果不存在则返回None
        """
        normalized = self._normalize_text(text)
        
        # 直接匹配
        if normalized in self.entity_library:
            return self.entity_library[normalized]
        
        # 同义词匹配
        if normalized in self.synonyms:
            canonical = self.synonyms[normalized]
            if canonical in self.entity_library:
                return self.entity_library[canonical]
        
        # 模糊匹配
        for entity_text, entity in self.entity_library.items():
            if normalized in entity_text or entity_text in normalized:
                return entity
        
        return None
    
    def find_related_entities(self, entity_text: str, relation_type: Optional[RelationType] = None) -> List[Tuple[Entity, RelationType, Entity]]:
        """查找与实体相关的其他实体"""
        normalized = self._normalize_text(entity_text)
        results = []
        
        if normalized in self.entity_library:
            entity = self.entity_library[normalized]
            if entity.id in self.relation_library:
                for relation in self.relation_library[entity.id]:
                    if relation_type is None or relation.predicate == relation_type:
                        results.append((relation.subject, relation.predicate, relation.object))
        
        return results
    
    def add_relation(self, relation: Relation):
        """添加关系到Ontology"""
        subj_id = relation.subject.id
        self.relation_library[subj_id].append(relation)
    
    def get_statistics(self) -> Dict:
        """获取Ontology统计信息"""
        return {
            "total_entities": len(self.entity_library),
            "total_relations": sum(len(rels) for rels in self.relation_library.values()),
            "entities_by_type": {t.value: len(entities) for t, entities in self.type_index.items()},
            "custom_types": list(self.custom_types.keys()),
            "synonyms_count": len(self.synonyms)
        }
    
    def _normalize_text(self, text: str) -> str:
        """标准化文本"""
        return text.lower().strip()


class BaseNERModel:
    """NER模型基类"""
    
    def __init__(self, language: str = "en"):
        self.language = language
        self.is_loaded = False
    
    def load(self):
        """加载模型"""
        raise NotImplementedError
    
    def extract(self, text: str) -> List[Entity]:
        """提取实体"""
        raise NotImplementedError
    
    def detect_language(self, text: str) -> str:
        """检测语言"""
        # 简单启发式检测
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        total_chars = len(text.replace(" ", ""))
        if total_chars > 0 and chinese_chars / total_chars > 0.3:
            return "zh"
        return "en"


class ChineseNERModel(BaseNERModel):
    """
    中文NER模型
    使用jieba分词 + 规则/统计方法
    可选：加载BERT模型
    """
    
    def __init__(self, use_bert: bool = False, bert_model: str = "bert-base-chinese"):
        super().__init__("zh")
        self.use_bert = use_bert
        self.bert_model_name = bert_model
        self.jieba = None
        self.bert_tokenizer = None
        self.bert_model = None
        
        # 规则模式
        self.patterns = {
            EntityType.PERSON: [
                r'[\u4e00-\u9fff]{2,4}(?:先生|女士|教授|博士|医生|老师)',
                r'(?:张|王|李|赵|刘|陈|杨|黄|吴|周|徐|孙|马|朱|胡|郭|林|何|高|罗)[\u4e00-\u9fff]{1,2}',
            ],
            EntityType.PLACE: [
                r'(?:北京|上海|广州|深圳|杭州|南京|成都|重庆|武汉|西安|天津|苏州|郑州|长沙|沈阳|青岛|宁波|东莞|无锡)[市]?',
                r'(?:中国|美国|日本|德国|法国|英国|意大利|加拿大|澳大利亚|俄罗斯|印度|巴西|韩国|西班牙|墨西哥|印度尼西亚|荷兰|沙特阿拉伯|土耳其|瑞士|波兰|比利时|瑞典|阿根廷|泰国|奥地利|挪威|阿联酋|以色列|丹麦|马来西亚|新加坡|爱尔兰|南非|菲律宾|埃及|孟加拉国|越南|智利|芬兰|巴基斯坦|罗马尼亚|捷克|葡萄牙|新西兰|希腊|伊拉克|阿尔及利亚|卡塔尔|哈萨克斯坦|匈牙利|科威特)[国]?',
            ],
            EntityType.ORGANIZATION: [
                r'(?:[^\s]{2,10})(?:公司|集团|银行|大学|学院|医院|研究所|中心|协会|学会|基金会|报社|电视台)',
            ],
            EntityType.TIME: [
                r'\d{4}年(?:\d{1,2}月)?(?:\d{1,2}[日号])?',
                r'(?:19|20)\d{2}(?:[-/]\d{1,2}(?:[-/]\d{1,2})?)?',
                r'(?:上午|下午|晚上|早上)?\d{1,2}(?::\d{2})?(?:点|时)',
            ],
            EntityType.MONEY: [
                r'(?:\d+(?:,\d{3})*(?:\.\d{1,2})?\s*(?:元|人民币|美元|美金|欧元|英镑|日元|韩元))',
                r'(?:\d+(?:\.\d{1,2})?\s*[万亿]?(?:元|人民币|美元))',
            ],
        }
    
    def load(self):
        """加载模型"""
        import jieba
        import jieba.posseg as pseg
        self.jieba = jieba
        self.pseg = pseg
        
        # 加载用户词典
        user_dict_path = Path("user_dict.txt")
        if user_dict_path.exists():
            self.jieba.load_userdict(str(user_dict_path))
        
        if self.use_bert:
            try:
                from transformers import BertTokenizer, BertForTokenClassification
                self.bert_tokenizer = BertTokenizer.from_pretrained(self.bert_model_name)
                self.bert_model = BertForTokenClassification.from_pretrained(self.bert_model_name)
                self.is_loaded = True
            except Exception as e:
                print(f"BERT model loading failed: {e}, falling back to rule-based")
                self.use_bert = False
        
        self.is_loaded = True
    
    def extract(self, text: str) -> List[Entity]:
        """提取中文实体"""
        if not self.is_loaded:
            self.load()
        
        entities = []
        
        # 1. 规则匹配
        for entity_type, patterns in self.patterns.items():
            for pattern in patterns:
                for match in re.finditer(pattern, text):
                    entity = Entity(
                        text=match.group(),
                        type=entity_type,
                        start=match.start(),
                        end=match.end(),
                        confidence=0.8,
                        metadata={"source": "rule"}
                    )
                    entities.append(entity)
        
        # 2. jieba词性标注辅助
        words = self.pseg.cut(text)
        for word, flag in words:
            if flag in ['nr', 'nr1', 'nr2', 'nrj', 'nrf']:  # 人名
                start = text.find(word)
                if start >= 0 and not any(e.start == start for e in entities):
                    entities.append(Entity(
                        text=word,
                        type=EntityType.PERSON,
                        start=start,
                        end=start + len(word),
                        confidence=0.7,
                        metadata={"source": "jieba", "flag": flag}
                    ))
            elif flag in ['ns', 'nsf']:  # 地名
                start = text.find(word)
                if start >= 0 and not any(e.start == start for e in entities):
                    entities.append(Entity(
                        text=word,
                        type=EntityType.PLACE,
                        start=start,
                        end=start + len(word),
                        confidence=0.7,
                        metadata={"source": "jieba", "flag": flag}
                    ))
            elif flag in ['nt', 'ntu', 'ntc', 'nto', 'ntf']:  # 机构名
                start = text.find(word)
                if start >= 0 and not any(e.start == start for e in entities):
                    entities.append(Entity(
                        text=word,
                        type=EntityType.ORGANIZATION,
                        start=start,
                        end=start + len(word),
                        confidence=0.7,
                        metadata={"source": "jieba", "flag": flag}
                    ))
        
        # 3. 如果使用BERT
        if self.use_bert and self.bert_model:
            bert_entities = self._extract_with_bert(text)
            entities.extend(bert_entities)
        
        # 去重和排序
        return self._deduplicate_entities(entities)
    
    def _extract_with_bert(self, text: str) -> List[Entity]:
        """使用BERT提取实体"""
        # 简化实现，实际使用时需要完整的NER pipeline
        entities = []
        # TODO: 实现BERT-based NER
        return entities
    
    def _deduplicate_entities(self, entities: List[Entity]) -> List[Entity]:
        """去重并排序实体"""
        seen = set()
        unique = []
        for e in sorted(entities, key=lambda x: x.confidence, reverse=True):
            key = (e.text, e.start, e.end)
            if key not in seen:
                seen.add(key)
                unique.append(e)
        return sorted(unique, key=lambda x: x.start)


class EnglishNERModel(BaseNERModel):
    """
    英文NER模型
    使用spaCy或transformers
    """
    
    def __init__(self, use_transformer: bool = True, model: str = "en_core_web_trf"):
        super().__init__("en")
        self.use_transformer = use_transformer
        self.model_name = model
        self.nlp = None
    
    def load(self):
        """加载spaCy模型"""
        try:
            import spacy
            if self.use_transformer:
                try:
                    self.nlp = spacy.load(self.model_name)
                except OSError:
                    print(f"Transformer model not found, using default...")
                    self.nlp = spacy.load("en_core_web_sm")
                    self.use_transformer = False
            else:
                self.nlp = spacy.load("en_core_web_sm")
            self.is_loaded = True
        except Exception as e:
            print(f"spaCy model loading failed: {e}")
            # 回退到简单规则
            self.nlp = None
            self.is_loaded = True
    
    def extract(self, text: str) -> List[Entity]:
        """提取英文实体"""
        if not self.is_loaded:
            self.load()
        
        entities = []
        
        if self.nlp:
            doc = self.nlp(text)
            
            # spaCy实体标签映射
            label_map = {
                "PERSON": EntityType.PERSON,
                "ORG": EntityType.ORGANIZATION,
                "GPE": EntityType.PLACE,
                "LOC": EntityType.PLACE,
                "PRODUCT": EntityType.PRODUCT,
                "EVENT": EntityType.EVENT,
                "WORK_OF_ART": EntityType.WORK_OF_ART,
                "DATE": EntityType.TIME,
                "TIME": EntityType.TIME,
                "MONEY": EntityType.MONEY,
                "FAC": EntityType.PLACE,
                "NORP": EntityType.CONCEPT,
            }
            
            for ent in doc.ents:
                entity_type = label_map.get(ent.label_, EntityType.CONCEPT)
                entities.append(Entity(
                    text=ent.text,
                    type=entity_type,
                    start=ent.start_char,
                    end=ent.end_char,
                    confidence=0.85,
                    metadata={"source": "spacy", "label": ent.label_}
                ))
        else:
            # 回退到简单规则
            entities = self._rule_based_extraction(text)
        
        return entities
    
    def _rule_based_extraction(self, text: str) -> List[Entity]:
        """基于规则的简单英文NER"""
        entities = []
        
        # 大写词可能是专有名词
        pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
        for match in re.finditer(pattern, text):
            entities.append(Entity(
                text=match.group(),
                type=EntityType.CONCEPT,
                start=match.start(),
                end=match.end(),
                confidence=0.5,
                metadata={"source": "rule"}
            ))
        
        return entities


class RelationExtractor:
    """
    关系抽取器
    使用BERT进行关系分类
    """
    
    def __init__(self, model_name: str = "distilbert-base-uncased"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.label_map = {i: rel for i, rel in enumerate(RelationType)}
        self.is_loaded = False
    
    def load(self):
        """加载关系抽取模型"""
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name, 
                num_labels=len(RelationType)
            )
            self.is_loaded = True
        except Exception as e:
            print(f"Relation extraction model loading failed: {e}")
            self.is_loaded = True  # 仍标记为加载，使用回退方法
    
    def extract_relations(self, text: str, entities: List[Entity]) -> List[Relation]:
        """
        从文本中抽取实体间的关系
        
        Args:
            text: 原始文本
            entities: 已识别的实体列表
        
        Returns:
            关系列表
        """
        if not self.is_loaded:
            self.load()
        
        relations = []
        
        # 如果没有足够的实体，返回空列表
        if len(entities) < 2:
            return relations
        
        # 1. 基于规则的启发式方法
        relations.extend(self._rule_based_extraction(text, entities))
        
        # 2. 如果使用深度学习模型
        if self.model and self.tokenizer:
            deep_relations = self._deep_learning_extraction(text, entities)
            relations.extend(deep_relations)
        
        return self._deduplicate_relations(relations)
    
    def _rule_based_extraction(self, text: str, entities: List[Entity]) -> List[Relation]:
        """基于规则的关系抽取"""
        relations = []
        
        # 关系模式
        patterns = [
            # 位于
            (r'{}.*?(?:位于|在|from|in|at).*?{}', RelationType.LOCATED_AT),
            # 工作于
            (r'{}.*?(?:工作|任职|works?\s+(?:at|for)|employed\s+by).*?{}', RelationType.WORKS_FOR),
            # 创建
            (r'{}.*?(?:创建|创立|建立|founded|created|established)\s+(?:by)?.*?{}', RelationType.CREATED_BY),
            # 参与
            (r'{}.*?(?:参加|参与|attended|participated\s+in).*?{}', RelationType.PARTICIPATED_IN),
        ]
        
        for i, ent1 in enumerate(entities):
            for j, ent2 in enumerate(entities):
                if i != j:
                    for pattern, rel_type in patterns:
                        # 尝试两种顺序
                        regex1 = pattern.format(re.escape(ent1.text), re.escape(ent2.text))
                        regex2 = pattern.format(re.escape(ent2.text), re.escape(ent1.text))
                        
                        if re.search(regex1, text, re.IGNORECASE):
                            relations.append(Relation(
                                subject=ent1,
                                predicate=rel_type,
                                object=ent2,
                                confidence=0.6,
                                metadata={"source": "rule"}
                            ))
                        elif re.search(regex2, text, re.IGNORECASE):
                            relations.append(Relation(
                                subject=ent2,
                                predicate=rel_type,
                                object=ent1,
                                confidence=0.6,
                                metadata={"source": "rule"}
                            ))
        
        return relations
    
    def _deep_learning_extraction(self, text: str, entities: List[Entity]) -> List[Relation]:
        """使用深度学习模型抽取关系"""
        relations = []
        
        try:
            import torch
            
            for i, ent1 in enumerate(entities):
                for j, ent2 in enumerate(entities):
                    if i >= j:
                        continue
                    
                    # 构建输入: [CLS] 实体1 [SEP] 实体2 [SEP] 句子
                    marked_text = f"{ent1.text} [SEP] {ent2.text} [SEP] {text}"
                    
                    inputs = self.tokenizer(
                        marked_text,
                        return_tensors="pt",
                        truncation=True,
                        max_length=512
                    )
                    
                    with torch.no_grad():
                        outputs = self.model(**inputs)
                        probs = torch.softmax(outputs.logits, dim=1)
                        pred = torch.argmax(probs, dim=1).item()
                        confidence = probs[0][pred].item()
                    
                    # 过滤低置信度的预测
                    if confidence > 0.7 and pred > 0:  # 假设0是无关系
                        rel_type = self.label_map.get(pred, RelationType.RELATED_TO)
                        relations.append(Relation(
                            subject=ent1,
                            predicate=rel_type,
                            object=ent2,
                            confidence=confidence,
                            metadata={"source": "bert"}
                        ))
        
        except Exception as e:
            print(f"Deep learning extraction failed: {e}")
        
        return relations
    
    def _deduplicate_relations(self, relations: List[Relation]) -> List[Relation]:
        """去重关系"""
        seen = set()
        unique = []
        for r in sorted(relations, key=lambda x: x.confidence, reverse=True):
            key = (r.subject.text, r.predicate.value, r.object.text)
            if key not in seen:
                seen.add(key)
                unique.append(r)
        return unique


class EnhancedNER:
    """
    增强版NER服务
    整合多种模型和Ontology管理
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        
        # 初始化模型
        self.chinese_model = ChineseNERModel(
            use_bert=self.config.get("use_bert_chinese", False),
            bert_model=self.config.get("chinese_bert_model", "bert-base-chinese")
        )
        
        self.english_model = EnglishNERModel(
            use_transformer=self.config.get("use_transformer", True),
            model=self.config.get("english_model", "en_core_web_trf")
        )
        
        # 关系抽取器
        self.relation_extractor = RelationExtractor(
            model_name=self.config.get("relation_model", "distilbert-base-uncased")
        )
        
        # Ontology管理器
        self.ontology = OntologyManager(
            storage_path=self.config.get("ontology_path", "ontology_db")
        )
        
        # 是否启用Ontology增强
        self.use_ontology = self.config.get("use_ontology", True)
        
        self._models_loaded = False
    
    def load_models(self):
        """加载所有模型"""
        if self._models_loaded:
            return
        
        print("Loading NER models...")
        self.chinese_model.load()
        self.english_model.load()
        self.relation_extractor.load()
        self._models_loaded = True
        print("NER models loaded successfully")
    
    def extract(self, text: str, extract_relations: bool = True, use_ontology: Optional[bool] = None) -> NERResult:
        """
        执行NER提取
        
        Args:
            text: 输入文本
            extract_relations: 是否抽取关系
            use_ontology: 是否使用Ontology增强
        
        Returns:
            NER结果
        """
        if not self._models_loaded:
            self.load_models()
        
        use_ontology = use_ontology if use_ontology is not None else self.use_ontology
        
        # 检测语言
        language = self._detect_language(text)
        
        # 选择模型
        if language == "zh":
            entities = self.chinese_model.extract(text)
        else:
            entities = self.english_model.extract(text)
        
        # Ontology增强
        if use_ontology:
            entities = self._enhance_with_ontology(entities)
        
        # 关系抽取
        relations = []
        if extract_relations:
            relations = self.relation_extractor.extract_relations(text, entities)
        
        return NERResult(
            text=text,
            entities=entities,
            relations=relations,
            language=language
        )
    
    def extract_batch(self, texts: List[str], extract_relations: bool = False) -> List[NERResult]:
        """批量提取"""
        return [self.extract(text, extract_relations) for text in texts]
    
    def _detect_language(self, text: str) -> str:
        """检测文本语言"""
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        total_chars = len(text.replace(" ", ""))
        if total_chars > 0 and chinese_chars / total_chars > 0.3:
            return "zh"
        return "en"
    
    def _enhance_with_ontology(self, entities: List[Entity]) -> List[Entity]:
        """使用Ontology增强实体"""
        enhanced = []
        for entity in entities:
            # 尝试链接到Ontology
            linked = self.ontology.link_entity(entity.text)
            if linked:
                # 合并信息
                entity.id = linked.id
                entity.confidence = max(entity.confidence, linked.confidence)
                entity.metadata.update(linked.metadata)
                entity.metadata["linked_to_ontology"] = True
            else:
                # 添加到Ontology
                self.ontology.add_entity(entity)
            enhanced.append(entity)
        
        return enhanced
    
    def add_to_ontology(self, text: str, entity_type: str, metadata: Optional[Dict] = None):
        """手动添加实体到Ontology"""
        entity = Entity(
            text=text,
            type=EntityType(entity_type),
            start=0,
            end=len(text),
            metadata=metadata or {}
        )
        self.ontology.add_entity(entity)
        self.ontology.save_data()
    
    def add_synonym(self, synonym: str, canonical: str):
        """添加同义词"""
        self.ontology.add_synonym(synonym, canonical)
        self.ontology.save_data()
    
    def get_entity_info(self, text: str) -> Optional[Dict]:
        """获取实体详细信息"""
        entity = self.ontology.link_entity(text)
        if entity:
            related = self.ontology.find_related_entities(text)
            return {
                "entity": entity.to_dict(),
                "related": [
                    {
                        "subject": s.to_dict(),
                        "predicate": p.value,
                        "object": o.to_dict()
                    }
                    for s, p, o in related
                ]
            }
        return None
    
    def get_ontology_stats(self) -> Dict:
        """获取Ontology统计"""
        return self.ontology.get_statistics()
    
    def save_ontology(self):
        """保存Ontology到磁盘"""
        self.ontology.save_data()


# 便捷函数
def create_ner_service(config: Optional[Dict] = None) -> EnhancedNER:
    """创建NER服务实例"""
    return EnhancedNER(config)


def quick_extract(text: str, language: str = "auto") -> Dict:
    """快速提取实体（无需配置）"""
    ner = EnhancedNER()
    result = ner.extract(text)
    return result.to_dict()


if __name__ == "__main__":
    # 测试
    ner = EnhancedNER()
    
    # 中文测试
    chinese_text = "马化腾是腾讯公司的创始人，公司总部位于深圳。"
    print("\n中文测试:")
    print(f"文本: {chinese_text}")
    result = ner.extract(chinese_text)
    print(f"语言: {result.language}")
    print(f"实体: {[e.to_dict() for e in result.entities]}")
    print(f"关系: {[r.to_dict() for r in result.relations]}")
    
    # 英文测试
    english_text = "Elon Musk founded SpaceX in California. The company is located in Hawthorne."
    print("\n英文测试:")
    print(f"Text: {english_text}")
    result = ner.extract(english_text)
    print(f"Language: {result.language}")
    print(f"Entities: {[e.to_dict() for e in result.entities]}")
    print(f"Relations: {[r.to_dict() for r in result.relations]}")
    
    # Ontology统计
    print("\nOntology统计:")
    print(ner.get_ontology_stats())
    
    # 保存
    ner.save_ontology()
