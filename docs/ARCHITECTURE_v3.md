# POS v3.0 系统架构

## 整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                          用户层                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │   Web界面   │  │   移动端    │  │  语音助手   │              │
│  │  (React)    │  │  (Future)   │  │  (Future)   │              │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘              │
└─────────┼────────────────┼────────────────┼─────────────────────┘
          │                │                │
          └────────────────┴────────────────┘
                           │ HTTPS/HTTP2
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                       API网关层 (Nginx)                          │
│              负载均衡 / SSL终结 / 静态资源缓存                    │
└─────────────────────────────────────────────────────────────────┘
                           │
          ┌────────────────┼────────────────┐
          │                │                │
          ▼                ▼                ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│   查询服务      │ │   写入服务      │ │   推理服务      │
│   (Read API)    │ │   (Write API)   │ │ (Async Worker)  │
└────────┬────────┘ └────────┬────────┘ └────────┬────────┘
         │                   │                   │
         └───────────────────┼───────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                     POS Core (C++) 服务层                        │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │  Core Engine                                                ││
│  │  ├── Input Parser (NER + Time + Location)                   ││
│  │  ├── Ontology Manager (Concept CRUD + Linking)              ││
│  │  ├── Memory Store (Vector + Graph + Temporal)               ││
│  │  ├── Reasoning Engine (Rules + Patterns)                    ││
│  │  └── Recommendation Engine (Multi-strategy)                 ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
          │                   │                   │
          ▼                   ▼                   ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│   本体存储      │ │   记忆存储      │ │   向量存储      │
│  (RocksDB)      │ │  (RocksDB)      │ │ (sqlite-vec/    │
│  概念+关系      │ │  原始内容       │ │  Milvus)        │
└─────────────────┘ └─────────────────┘ └─────────────────┘
          │                   │                   │
          └───────────────────┼───────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      ML服务层 (Python)                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │  NER服务    │  │  Embedding  │  │  LLM服务    │              │
│  │  (BERT-CRF) │  │  (384/1024) │  │ (GPT/Local) │              │
│  └─────────────┘  └─────────────┘  └─────────────┘              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │ 关系抽取    │  │ 情感分析    │  │ 文本生成    │              │
│  │ (BERT-RE)   │  │ (Sentiment) │  │ (T5/GPT)    │              │
│  └─────────────┘  └─────────────┘  └─────────────┘              │
└─────────────────────────────────────────────────────────────────┘
```

## 核心组件详解

### 1. 输入解析层 (Input Parser)

```
用户输入: "明天晚上和女朋友在海底捞过生日"
    │
    ▼
┌─────────────────────────────────────┐
│  预处理 (Tokenization)               │
│  - 分词                              │
│  - 词性标注                          │
│  - 依存句法分析                      │
└────────────────┬────────────────────┘
                 │
    ┌────────────┼────────────┐
    ▼            ▼            ▼
┌───────┐   ┌───────┐   ┌───────┐
│  NER  │   │ 时间  │   │ 地点  │
│ 抽取  │   │ 解析  │   │ 解析  │
└───┬───┘   └───┬───┘   └───┬───┘
    │           │           │
    ▼           ▼           ▼
女朋友/PERSON  2024-01-17T19:00  海底捞/PLACE

┌─────────────────────────────────────┐
│  关系抽取                            │
│  - 主谓宾关系                        │
│  - 修饰关系                          │
│  - 时空关系                          │
└─────────────────────────────────────┘
```

### 2. 存储层 (Storage Layer)

#### 2.1 本体存储 (Ontology Store)
```cpp
// RocksDB Schema
concept:{id} -> {
  "id": "person_zhangsan",
  "type": "PERSON", 
  "label": "张三",
  "aliases": ["老张", "张三"],
  "properties": {
    "birth_date": "1990-01-01",
    "occupation": "工程师"
  },
  "relations": [
    {"type": "KNOWS", "target": "person_lisi", "weight": 0.9}
  ],
  "vector": [0.1, 0.2, ...],  // 概念向量
  "created_at": "2024-01-01",
  "updated_at": "2024-01-15"
}

// 索引
index:label:{label} -> [concept_ids]
index:type:{type} -> [concept_ids]
```

#### 2.2 记忆存储 (Memory Store)
```cpp
memory:{id} -> {
  "id": "mem_abc123",
  "content": "原始文本",
  "content_vector": [0.1, 0.2, ...],  // 语义向量
  "timestamp": "2024-01-15T19:00:00",
  "location": {"lat": 39.9, "lng": 116.4},
  "entities": [
    {"text": "张三", "concept_id": "person_zhangsan", "offset": 0}
  ],
  "relations": [
    {"subject": "张三", "predicate": "LOCATED_AT", "object": "海底捞"}
  ],
  "sentiment": {"valence": 0.8, "arousal": 0.6},
  "source": "user_input",
  "created_at": "2024-01-15T19:00:00"
}

// 多维度索引
vector_index: HNSW(384维)
temporal_index: 时间戳排序
spatial_index: RTree地理索引
```

#### 2.3 向量存储 (Vector Store)
使用 sqlite-vec:
```sql
-- 创建虚拟表
CREATE VIRTUAL TABLE vec_memories USING vec0(
  memory_id TEXT PRIMARY KEY,
  embedding FLOAT[384]  -- 向量维度
);

-- 向量相似度搜索
SELECT memory_id, distance
FROM vec_memories
WHERE embedding MATCH ?
ORDER BY distance
LIMIT 10;
```

### 3. 推理引擎 (Reasoning Engine)

#### 3.1 规则系统
```cpp
// 规则定义
struct InferenceRule {
    string id;
    string name;
    RuleType type;
    
    // 条件函数
    function<bool(Context)> condition;
    
    // 动作函数
    function<void(Context)> action;
    
    // 元数据
    float confidence;
    int priority;
};

// 内置规则
1. friend_of_friend (传递性)
   IF: A knows B AND B knows C
   THEN: A potentially knows C (confidence: 0.7)

2. schedule_conflict (冲突检测)
   IF: time_overlap(mem1, mem2) AND 
       distance(mem1.location, mem2.location) > 1km
   THEN: conflict detected (severity: high)

3. habit_pattern (习惯识别)
   IF: count(memories with concept X) >= 3 AND
       temporal_pattern detected
   THEN: habit identified (confidence: frequency-based)
```

#### 3.2 模式发现算法
```cpp
// Apriori算法挖掘频繁模式
vector<Pattern> discoverPatterns(int min_support) {
    // 1. 生成频繁项集
    auto frequent_1 = findFrequent1Itemsets(min_support);
    
    // 2. 迭代生成k-项集
    for (k = 2; frequent_{k-1} not empty; k++) {
        auto candidates = aprioriGen(frequent_{k-1});
        auto frequent_k = filterBySupport(candidates, min_support);
    }
    
    // 3. 生成关联规则
    return generateRules(frequent_itemsets);
}
```

### 4. 推荐引擎 (Recommendation Engine)

#### 4.1 多策略融合
```cpp
class HybridRecommender {
    vector<Recommendation> recommend(Context ctx) {
        vector<Recommendation> all_recs;
        
        // 1. 基于时间的推荐
        auto time_recs = timeBasedRecommender.recommend(ctx);
        all_recs.insert(all_recs.end(), time_recs.begin(), time_recs.end());
        
        // 2. 基于位置的推荐
        auto loc_recs = locationBasedRecommender.recommend(ctx);
        all_recs.insert(all_recs.end(), loc_recs.begin(), loc_recs.end());
        
        // 3. 基于社交的推荐
        auto social_recs = socialBasedRecommender.recommend(ctx);
        all_recs.insert(all_recs.end(), social_recs.begin(), social_recs.end());
        
        // 4. 协同过滤
        auto cf_recs = collaborativeFiltering.recommend(ctx);
        all_recs.insert(all_recs.end(), cf_recs.begin(), cf_recs.end());
        
        // 5. 融合排序
        return fuseAndRank(all_recs);
    }
};
```

#### 4.2 融合排序算法
```cpp
// 加权融合
float score(Recommendation rec, Context ctx) {
    float score = 0;
    
    // 置信度权重
    score += rec.confidence * 0.3;
    
    // 时效性权重
    score += rec.timeliness * 0.2;
    
    // 个性化权重
    score += rec.personalization * 0.3;
    
    // 多样性权重
    score += rec.diversity * 0.2;
    
    return score;
}
```

## 数据流

### 写入流程 (Write Path)
```
用户输入
    │
    ▼
┌─────────────┐
│ 输入解析     │
│ (NER+Time)  │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ 实体消歧     │
│ (Disambig)  │
└──────┬──────┘
       │
       ▼
┌─────────────┐     ┌─────────────┐
│ 本体更新     │────→│ 概念存储     │
│ (Ontology)  │     │ (RocksDB)   │
└──────┬──────┘     └─────────────┘
       │
       ▼
┌─────────────┐     ┌─────────────┐
│ 记忆创建     │────→│ 记忆存储     │
│ (Memory)    │     │ (RocksDB+   │
└──────┬──────┘     │  HNSW)      │
       │            └─────────────┘
       ▼
┌─────────────┐
│ 触发推理     │
│ (Reasoning) │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ 生成推荐     │
│ (Recommend) │
└─────────────┘
```

### 读取流程 (Read Path)
```
用户查询
    │
    ▼
┌─────────────┐
│ 查询解析     │
│ (Query      │
│  Parser)    │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ 策略选择     │
│ (Strategy   │
│  Selection) │
└──────┬──────┘
       │
       ├─── 语义查询 ──→ 向量检索 (HNSW)
       │
       ├─── 概念查询 ──→ 本体遍历 (Graph)
       │
       ├─── 时间查询 ──→ 时间索引 (Temporal)
       │
       └─── 空间查询 ──→ 空间索引 (RTree)
       │
       ▼
┌─────────────┐
│ 结果融合     │
│ (Fusion)    │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ 推理增强     │
│ (Inference) │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ 返回结果     │
│ (Response)  │
└─────────────┘
```

## 性能优化

### 1. 缓存策略
```
L1缓存: 内存 (Redis/Memcached)
  - 热点数据
  - 用户会话

L2缓存: 本地磁盘
  - 向量索引缓存
  - 查询结果缓存

L3存储: RocksDB
  - 持久化数据
```

### 2. 索引优化
```cpp
// 复合索引
index:concept:memories:{concept_id}:{timestamp} -> [memory_ids]
index:location:{geohash}:{timestamp} -> [memory_ids]

// 预计算
预计算常用查询模式
预计算关系强度
预计算习惯模式
```

### 3. 异步处理
```cpp
// 写入异步化
void processInputAsync(Input input) {
    // 同步：解析和存储
    auto result = parseAndStore(input);
    
    // 异步：推理和推荐
    async_execute([result]() {
        auto inferences = reasoningEngine.infer(result);
        auto recommendations = recommender.generate(inferences);
        notificationService.notify(recommendations);
    });
}
```

## 安全与隐私

### 1. 数据加密
```
传输加密: TLS 1.3
存储加密: AES-256-GCM
密钥管理: HashiCorp Vault / AWS KMS
```

### 2. 访问控制
```
认证: JWT Token
授权: RBAC (Role-Based Access Control)
审计: 操作日志记录
```

### 3. 隐私保护
```
数据脱敏: 敏感信息打码
本地优先: 核心数据不上云
差分隐私: 统计分析加噪
```

## 监控与运维

### 1. 指标监控
```
业务指标:
- QPS (Queries Per Second)
- 延迟 P50/P95/P99
- 错误率
- 推理准确率

系统指标:
- CPU/Memory/Disk
- 网络IO
- 数据库连接数
```

### 2. 日志系统
```
结构化日志 (JSON格式)
分布式追踪 (OpenTelemetry)
错误上报 (Sentry)
```

### 3. 告警机制
```
告警级别: P0(紧急) / P1(重要) / P2(一般)
通知渠道: 钉钉/飞书/邮件/短信
告警策略: 阈值/异常检测/趋势预测
```
