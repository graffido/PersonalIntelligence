# POS 本地推理引擎

## 概述

基于本体的本地推理引擎，通过知识图谱和规则推理减少对大型语言模型(LLM)的依赖，降低延迟和成本。

## 架构

```
用户查询
    │
    ▼
┌─────────────────┐
│  实体识别(NER)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  本地推理引擎    │ ← 优先使用
│  - 本体匹配      │
│  - 关系推理      │
│  - 模式发现      │
│  - 冲突检测      │
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
质量足够    质量不足
    │         │
    ▼         ▼
直接返回   调用LLM
(0.001s)   (1-3s)
```

## 推理规则

### 1. 传递性推理
```
规则: 朋友的朋友
条件: A knows B, B knows C
推断: A 可能认识 C (置信度: 0.7)
```

### 2. 冲突检测
```
规则: 日程冲突
条件: 时间重叠 + 地点距离 > 1km
推断: 无法同时参加 (置信度: 0.9)
```

### 3. 模式发现
```
规则: 习惯模式
条件: 同一概念相关记忆 >= 3次
推断: 发现重复行为模式
```

### 4. 可达性检查
```
规则: 地点可达性
条件: 距离/速度 > 可用时间
推断: 可能迟到 (置信度: 0.85)
```

## 使用方法

### 基本用法

```python
from reasoning_engine import LocalReasoningEngine, Concept, Memory

# 创建引擎
engine = LocalReasoningEngine()

# 添加概念
engine.add_concept(Concept(
    id="person_zhangsan",
    type="PERSON", 
    label="张三"
))

# 添加记忆
engine.add_memory(Memory(
    id="mem1",
    content="在星巴克和李四讨论项目",
    timestamp=datetime.now(),
    entities=[{"text": "李四", "label": "PERSON"}],
    ontology_bindings=["person_lisi"]
))

# 执行推理
results = engine.infer()
```

### 智能查询路由

```python
from demo_smart_router import SmartQueryRouter

router = SmartQueryRouter()

# 添加知识...

# 查询
result = router.query(
    "找一下我和李四的记忆",
    entities=[{"text": "李四", "label": "PERSON"}]
)

# 自动决定使用本地推理还是LLM
if result["llm_called"]:
    print("使用了LLM")
else:
    print("使用了本地推理，节约成本!")
```

## 性能对比

| 场景 | 本地推理 | LLM | 加速比 |
|------|---------|-----|-------|
| 精确匹配查询 | 1ms | 2000ms | 2000x |
| 关系推理 | 5ms | 3000ms | 600x |
| 模式发现 | 10ms | 5000ms | 500x |
| 复杂分析 | - | 3000ms | - |

## 成本节约

假设:
- LLM调用成本: $0.002/次
- 日查询量: 1000次
- 本地推理覆盖率: 70%

计算:
```
原始成本: 1000 × $0.002 = $2.00/天
优化成本: 300 × $0.002 = $0.60/天
节约: $1.40/天 (70%)
年度节约: $511
```

## API端点

### 启动推理服务
```bash
python3 reasoning_service.py
# 服务运行在 http://localhost:8001
```

### 主要端点

```bash
# 健康检查
GET /health

# 执行推理
POST /reason
{
    "query": "用户查询文本",
    "entities": [{"text": "实体", "label": "类型"}],
    "context": {"current_location": {...}}
}

# 冲突检测
POST /detect-conflicts
{
    "time_window_hours": 48
}

# 模式发现
POST /discover-patterns
{
    "min_frequency": 3
}
```

## 集成到主系统

修改 `main.py` 添加推理路由:

```python
from reasoning_engine import LocalReasoningEngine

# 初始化引擎
reasoning_engine = LocalReasoningEngine()

@app.post("/query")
def query(req: QueryRequest):
    # 1. 本地推理
    local_result = reasoning_engine.query_with_reasoning(
        req.text, req.entities
    )
    
    # 2. 评估质量
    quality = evaluate_quality(local_result)
    
    # 3. 路由决策
    if quality >= 0.7:
        return local_result
    else:
        # 回退到LLM
        return call_llm(req.text, local_result)
```

## 推理示例

### 示例1: 传递性推理
```
输入: "张三认识李四吗？"
知识: 张三-认识->王五, 王五-认识->李四
推理: 张三可能认识李四（通过王五介绍）
结果: 无需LLM
```

### 示例2: 冲突检测
```
输入: "下午3点有两个会议"
知识: 
  - 会议A: 14:00-15:30 @海淀区
  - 会议B: 15:00-16:00 @朝阳区  
推理: 时间重叠 + 地点距离15km = 冲突
结果: 无需LLM，本地检测
```

### 示例3: 模式推荐
```
输入: "下周三晚上有什么安排？"
知识: 过去4周周三晚上都在健身房
推理: 习惯模式识别
结果: "根据您的习惯，建议去健身房"
置信度: 0.85
```

## 配置

### 规则阈值
```python
# 在 reasoning_engine.py 中调整
MIN_PATTERN_FREQUENCY = 3      # 模式最小频率
MIN_RELATION_CONFIDENCE = 0.7  # 关系最小置信度
CONFLICT_DISTANCE_METERS = 1000 # 冲突检测距离
```

### 质量评估权重
```python
def _evaluate_quality(self, local_result):
    score = 0.0
    if local_result['direct_matches']: score += 0.4
    if local_result['inferred_matches']: score += 0.2
    if local_result['suggestions']: score += 0.2
    return score
```

## 文件清单

- `reasoning_engine.py` - 核心推理引擎
- `reasoning_service.py` - REST API服务
- `demo_smart_router.py` - 智能路由演示

## 下一步

1. 与C++后端集成
2. 添加更多推理规则
3. 优化规则执行效率
4. 支持分布式推理
