# POS (Personal Ontology System) v2.0

## 项目概述

POS是一个**个人本体记忆系统**，基于C++构建高性能后端核心，实现统一输入处理、本体管理、本地推理和智能推荐。

## 核心架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        前端层 (Web UI)                           │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │  • 统一输入框 (自然语言)                                      ││
│  │  • 实体可视化展示                                            ││
│  │  • 推理结果展示                                              ││
│  │  • 推荐卡片                                                  ││
│  └─────────────────────────────────────────────────────────────┘│
└───────────────────────────┬─────────────────────────────────────┘
                            │ HTTP API
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                     POS Core 后端 (C++)                          │
│                      Port: 9000                                  │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │   API Server    │  │  Core Engine    │  │   ML Client     │  │
│  │   (Crow)        │←→│  (Business      │←→│  (HTTP/REST)    │  │
│  │                 │  │   Logic)        │  │                 │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │  Input Parser   │  │  Reasoning      │  │  Recommendation │  │
│  │  (NER+Time)     │  │  Engine         │  │  Engine         │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │  OntologyGraph  │  │  MemoryStore    │  │  Spatiotemporal │  │
│  │  (RocksDB)      │  │  (RocksDB+HNSW) │  │  Index (RTree)  │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ML Service (Python)                           │
│                      Port: 8000                                  │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │  NER/Entity     │  │  Embedding      │  │  Text Gen       │  │
│  │  Extraction     │  │  (384-dim)      │  │  (Optional)     │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## 核心能力

### 1. 统一输入处理 (Unified Input)

**单一输入框，自动提取所有信息：**

| 提取类型 | 示例 | 输出 |
|---------|------|------|
| **时间** | 今天、明天、早晨8点、下周一 | 标准化时间戳 |
| **地点** | 星巴克、海底捞、公司、医院 | PLACE概念 |
| **人物** | 中伟、李四、女朋友、同事 | PERSON概念 |
| **事件** | 开会、吃饭、锻炼、看病 | EVENT类型 |
| **物品** | 咖啡、电脑、文件 | OBJECT类型 |
| **关系** | 和...一起、在...见面 | 关系三元组 |

**处理流程：**
```
用户输入 → NER提取 → 消歧规范化 → 去重 → 创建概念 → 绑定记忆 → 执行推理 → 生成推荐
```

### 2. 本体管理 (Ontology Management)

**核心特性：**
- ✅ **自动概念创建**：从输入自动创建本体概念
- ✅ **实体消歧**：别名映射 (中伟 = 张伟 = 张中伟)
- ✅ **去重机制**：基于相似度和位置去重
- ✅ **双向绑定**：记忆 ↔ 概念双向关联
- ✅ **关系管理**：支持多种关系类型 (KNOWS, LOCATED_AT, PARTICIPATES_IN等)

**存储结构：**
```cpp
struct OntologyConcept {
    ConceptId id;                    // 唯一标识
    ConceptType type;                // PERSON/PLACE/EVENT/...
    string label;                    // 显示名称
    vector<Relation> relations;      // 关系列表
    set<MemoryId> bound_memories;    // 绑定的记忆
    float confidence;                // 置信度
};
```

### 3. 本地推理引擎 (Reasoning Engine)

**6大推理规则：**

| 规则类型 | 功能 | 示例 |
|---------|------|------|
| **传递性推理** | 朋友的朋友 → 潜在联系人 | 张三认识王五，王五认识李四 → 张三可能认识李四 |
| **冲突检测** | 时间/空间冲突发现 | 两个会议时间重叠且相距 >1km |
| **可达性检查** | 迟到预警 | 距离/速度 > 可用时间 → 可能迟到 |
| **模式识别** | 发现重复行为 | 每周三晚上去健身房 |
| **情感一致性** | 检测情绪与活动匹配 | 聚会时情绪低落 → 可能需要支持 |
| **关系强度** | 基于互动频率计算 | 张三互动5次/月 → 强关系 |

**推理结果类型：**
- `new_relation`：发现新关系
- `conflict`：检测到冲突
- `suggestion`：生成建议
- `pattern`：发现模式

### 4. 智能推荐系统 (Recommendation)

**6类推荐：**

```cpp
enum class RecommendationType {
    TIME_BASED,      // 时间推荐："根据习惯，建议早晨去星巴克"
    LOCATION_BASED,  // 位置推荐："您经常在这里见李四"
    SOCIAL_BASED,    // 社交推荐："已经30天没联系张三了"
    HABIT_BASED,     // 习惯推荐："您已连续4次周三健身"
    PREDICTIVE,      // 预测推荐："预计明天下午可能开会"
    CONTEXTUAL       // 情境推荐："日程冲突警告"
};
```

**推荐质量评估：**
- 置信度计算：基于历史频率和时间衰减
- 优先级排序：紧急冲突 > 习惯 > 社交 > 预测
- 个性化：基于个人历史模式

### 5. 预测能力 (Prediction)

**事件预测：**
- 基于时间模式预测下次活动时间
- 基于地点模式预测可能去向
- 基于社交模式预测可能联系人

**预测输出：**
```json
{
  "event_type": "健身",
  "predicted_time": "2024-01-17T19:00:00",
  "confidence": 0.85,
  "location_probability": {"健身房": 0.9, "家中": 0.1},
  "person_probability": {"教练": 0.7, "朋友": 0.3}
}
```

## API接口

### 核心端点

| 方法 | 端点 | 功能 |
|------|------|------|
| POST | `/api/v1/input` | 统一输入处理 |
| POST | `/api/v1/query` | 智能查询 |
| POST | `/api/v1/recommendations` | 获取推荐 |
| GET  | `/api/v1/predictions` | 事件预测 |
| GET  | `/api/v1/stats` | 系统统计 |

### 输入处理示例

**请求：**
```bash
curl -X POST http://localhost:9000/api/v1/input \
  -H "Content-Type: application/json" \
  -d '{
    "text": "今天早晨8点在星巴克和中伟讨论项目方案"
  }'
```

**响应：**
```json
{
  "success": true,
  "memory_id": "mem_a1b2c3d4",
  "entities": [
    {"text": "今天", "label": "DATE", "normalized": "今天"},
    {"text": "8点", "label": "TIME", "normalized": "8:00"},
    {"text": "星巴克", "label": "PLACE", "normalized": "星巴克"},
    {"text": "中伟", "label": "PERSON", "normalized": "中伟"}
  ],
  "reasoning_results": [
    {
      "rule": "习惯模式",
      "type": "pattern",
      "description": "经常在早晨与'中伟'相关的活动",
      "confidence": 0.8
    }
  ],
  "recommendations": [
    {
      "type": "time_based",
      "title": "早晨习惯",
      "description": "您经常在早晨处理重要事务",
      "confidence": 0.75
    }
  ],
  "stats": {
    "new_concepts": 2,
    "linked_concepts": 1
  }
}
```

## 技术栈

### 后端核心 (C++)
- **HTTP Server**: Crow (C++ micro web framework)
- **存储**: RocksDB (持久化) + HNSW (向量索引) + RTree (空间索引)
- **JSON**: nlohmann/json
- **HTTP Client**: libcurl
- **构建**: CMake

### ML服务 (Python)
- **框架**: FastAPI
- **NLP**: 规则NER + 可选深度学习模型
- **向量**: 384维embedding

### 前端
- **纯HTML/JS**: 无框架依赖，快速加载
- **样式**: 原生CSS (渐变背景 + 卡片设计)

## 性能指标

| 指标 | 数值 | 说明 |
|------|------|------|
| 输入处理延迟 | < 100ms | 实体提取+存储+推理 |
| 查询响应时间 | < 50ms | 本地推理无需LLM |
| 并发处理能力 | 1000+ QPS | C++高性能后端 |
| 存储容量 | 100万+ 记忆 | RocksDB支持 |
| LLM调用减少 | 70%+ | 本地推理替代 |

## 项目结构

```
/Users/bingo/projects/personal_intelligence/
├── pos-cpp/                      # C++后端核心
│   ├── src/
│   │   ├── core/
│   │   │   ├── common/           # 基础类型定义
│   │   │   ├── ontology/         # 本体图谱 (RocksDB)
│   │   │   ├── memory/           # 记忆存储 (分层存储)
│   │   │   ├── temporal/         # 时空索引 (RTree)
│   │   │   ├── reasoning/        # 推理引擎
│   │   │   └── pos_core_engine.h # 核心引擎接口
│   │   ├── api/                  # HTTP API (Crow)
│   │   └── ml/                   # ML客户端
│   └── CMakeLists.txt
│
├── ml-service/                   # Python ML服务
│   ├── unified_api.py            # 统一API服务
│   ├── unified_input_parser.py   # 输入解析
│   ├── reasoning_engine.py       # 推理引擎
│   ├── prediction_recommendation_engine.py  # 预测推荐
│   └── main.py                   # 主服务入口
│
├── pos_core_backend.py           # Python后端原型 (等效C++)
├── pos_core_frontend.html        # 前端界面
│
└── docs/                         # 文档
    └── ARCHITECTURE.md           # 本文件
```

## 设计理念

### 1. 后端核心原则
- **POS Core是系统的唯一核心**
- 前端只负责展示，不参与业务逻辑
- 所有推理、推荐、预测都在后端完成

### 2. 本地优先
- 优先使用本地推理，减少LLM调用
- 70%+ 的查询通过本地推理解决
- LLM仅作为fallback用于复杂分析

### 3. 统一抽象
- 单一输入框处理所有类型输入
- 统一的概念模型 (人/事/物/时/空)
- 统一的推理接口

### 4. 双向增强
- 记忆增强本体：从具体经验抽象知识
- 本体引导记忆：结构化组织经验

## 部署方式

### 开发环境
```bash
# 1. 启动ML服务
cd ml-service && python3 main.py

# 2. 编译并启动C++后端
cd pos-cpp/build && cmake .. && make -j4
./pos_server

# 3. 启动前端
python3 -m http.server 8080
```

### 生产环境
```bash
# Docker部署 (待实现)
docker-compose up -d
```

## 未来规划

- [ ] 完整的C++后端编译和部署
- [ ] 分布式存储支持
- [ ] 更强大的NLP模型集成
- [ ] 移动端APP
- [ ] 数据导入导出 (日历/社交媒体)

## 作者

POS Team

## License

MIT
