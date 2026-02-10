# POS 四层记忆系统 v2.0 优化说明

## 概述

本次优化将原有的三层记忆系统扩展为认知科学启发的**四层分层记忆架构**，并集成 Kimi (Moonshot AI) 作为首选 LLM 提供商。

## 1. ML Service - Kimi (Moonshot AI) 集成

### 变更文件
- `ml-service/llm_service.py` - 新增 KimiProvider 类
- `ml-service/config.yaml` - 添加 Kimi 配置

### 主要特性
- **新增 `KimiProvider` 类**：支持 Moonshot AI API（kimi-k2.5 等模型）
- **OpenAI 兼容接口**：复用现有 openai 库，无缝对接
- **智能路由更新**：默认将 Kimi 作为中等和复杂任务的首选模型
- **工具调用支持**：支持 Kimi 的 function calling 能力
- **流式响应**：支持流式输出

### 使用方法
```python
# 设置环境变量
export KIMI_API_KEY="your-api-key"

# 使用 Kimi 进行对话
llm_service = create_llm_service("config.yaml")
response = llm_service.chat("你好，请介绍一下自己")
print(response.content)
```

### 支持的模型
- `kimi-k2.5` - Kimi Code，推荐用于复杂任务
- `kimi-latest` - 最新版本
- `moonshot-v1-8k/32k/128k` - 不同上下文长度

---

## 2. C++ Core - 四层分层记忆系统

### 架构图

```
┌─────────────────────────────────────────────────────────────┐
│  Layer 0: Sensory Memory (感知记忆)                          │
│  - 毫秒级生命周期 (~200-500ms)                               │
│  - 原始输入缓冲：视觉、听觉、文本预处理                        │
│  - 自动衰减，选择性注意进入工作记忆                            │
└─────────────────────────────────────────────────────────────┘
                              ↓ 注意机制
┌─────────────────────────────────────────────────────────────┐
│  Layer 1: Working Memory (工作记忆)                          │
│  - 秒级生命周期 (~30-60s)                                    │
│  - 当前任务上下文，注意力焦点                                  │
│  - 有限容量 (7±2 个信息块)                                    │
│  - 纯内存存储，超高速访问                                      │
│  - 支持信息分块 (Chunking)                                    │
└─────────────────────────────────────────────────────────────┘
                              ↓ 编码巩固
┌─────────────────────────────────────────────────────────────┐
│  Layer 2: Long-term Memory (长期记忆)                        │
│  - 持久存储，理论上无上限                                       │
│  - 显式记忆：情景记忆 + 语义记忆                                │
│  - 向量索引 (HNSW) + 时空索引 + 图关系索引                     │
│  - 磁盘存储 (RocksDB) + 压缩                                   │
└─────────────────────────────────────────────────────────────┘
                              ↓ 统计学习
┌─────────────────────────────────────────────────────────────┐
│  Layer 3: Parameter Memory (参数记忆)                        │
│  - 隐式记忆：程序性知识、统计模式                               │
│  - 模型权重、访问模式、用户偏好                                  │
│  - 神经网络风格存储，渐进式更新                                  │
│  - 支持在线学习和自适应调整                                     │
└─────────────────────────────────────────────────────────────┘
```

### 新增/修改文件

#### 头文件
- `pos-cpp/src/core/memory/hierarchical_memory.h` - 四层记忆系统头文件

#### 实现文件
- `pos-cpp/src/core/memory/hierarchical_memory_part1.cpp` - 构造函数与初始化
- `pos-cpp/src/core/memory/hierarchical_memory_part2.cpp` - Layer 0 & 1 实现
- `pos-cpp/src/core/memory/hierarchical_memory_part3.cpp` - Layer 2 & 3 实现
- `pos-cpp/src/core/memory/hierarchical_memory_part4.cpp` - 维护任务与统计

#### 测试文件
- `pos-cpp/tests/test_quad_layer_memory.cpp` - 四层记忆系统测试

#### 构建配置
- `pos-cpp/CMakeLists.txt` - 更新以支持新组件

### 核心特性

#### Layer 0: 感知记忆 (Sensory Memory)
```cpp
// 接收感官输入
auto id = store.sensoryInputText("用户输入的文本", attention_hint=80.0f);

// 获取注意力焦点
auto focus = store.getAttentionFocus();

// 手动触发注意机制
store.triggerAttention({id1, id2});
```

- **环形缓冲区**：预分配固定大小缓冲区
- **自动衰减**：基于时间的自动清理
- **注意力机制**：高权重内容自动进入工作记忆

#### Layer 1: 工作记忆 (Working Memory)
```cpp
// 加载记忆到工作记忆
store.loadToWorkingMemory(memory, set_focus=true);

// 获取当前焦点记忆
auto focused = store.getFocusedMemory();

// 复述记忆（增强激活）
store.rehearse(memory_id);

// 创建信息块
auto chunk_id = store.createChunk("任务相关记忆", {id1, id2, id3});
```

- **有限容量**：基于 Miller's Law (7±2)
- **激活水平**：动态计算记忆的激活度
- **信息分块**：支持将相关记忆组织成块
- **复述机制**：增强记忆的持久性

#### Layer 2: 长期记忆 (Long-term Memory)
```cpp
// 存储到长期记忆
auto id = store.storeLongTerm(memory);

// 语义搜索
auto results = store.semanticSearch(embedding, top_k=10);

// 时空查询
auto memories = store.spatiotemporalQuery(
    time_start, time_end, bounding_box, limit=100);

// 关联记忆
store.associate(id1, id2, strength=0.8);
```

- **HNSW 向量索引**：高效的近似最近邻搜索
- **RTree 空间索引**：地理范围查询
- **时间索引**：基于时间的记忆检索
- **关系存储**：记忆间的关联关系

#### Layer 3: 参数记忆 (Parameter Memory)
```cpp
// 获取用户偏好向量
auto prefs = store.getUserPreferences();

// 更新偏好（梯度下降）
store.updateUserPreferences(gradient, learning_rate=0.001);

// 获取访问模式
auto pattern = store.getAccessPattern();

// 存储参数
store.storeParameter("key", values);
```

- **用户偏好向量**：256维嵌入表示用户偏好
- **访问模式模型**：24小时时间分布 + 类别分布
- **在线学习**：支持实时更新参数
- **统计学习**：从访问历史中提取模式

### 层间转换

```cpp
// 感知 → 工作记忆（通过注意机制）
auto working_id = store.attend(sensory_id, "解释后的内容");

// 工作 → 长期记忆（巩固）
auto ltm_id = store.consolidate(working_id);

// 长期 → 工作记忆（回忆）
store.recallToWorking(ltm_id);

// 长期 → 参数记忆（学习）
store.learnFromExperiences();
```

### 自动维护

```cpp
// 配置自动维护
HierarchicalMemoryConfig config;
config.enable_auto_consolidation = true;
config.enable_forgetting = true;
config.maintenance_interval_ms = 60000;  // 每分钟

// 手动触发维护
auto result = store.runMaintenance();
// result.decayed_sensory    - 衰减的感知记忆数
// result.consolidated       - 巩固到LTM的记忆数
// result.forgotten          - 遗忘的记忆数
// result.learned_params     - 学习的参数数
```

### 统计和监控

```cpp
auto stats = store.getStatistics();
// stats.sensory_count       - 感知记忆数量
// stats.working_count       - 工作记忆数量
// stats.long_term_count     - 长期记忆数量
// stats.parameter_count     - 参数数量
// stats.cache_hit_rate      - 缓存命中率
// stats.avg_access_latency_ms - 平均访问延迟
```

---

## 3. 构建说明

### 依赖要求
- CMake >= 3.16
- C++20 编译器
- RocksDB
- yaml-cpp
- Boost.Geometry
- HNSWLib (通过 FetchContent 自动下载)

### macOS
```bash
brew install rocksdb yaml-cpp boost cmake

mkdir build && cd build
cmake ..
make -j$(sysctl -n hw.ncpu)
```

### Ubuntu
```bash
sudo apt-get update
sudo apt-get install -y librocksdb-dev libyaml-cpp-dev libboost-all-dev cmake

mkdir build && cd build
cmake ..
make -j$(nproc)
```

### 运行测试
```bash
# 四层记忆系统测试
./test_quad_layer_memory

# 完整测试套件
./test_full_system
```

---

## 4. 性能优化

### 缓存策略
- **LRU 缓存**：长期记忆使用 LRU 缓存提高热点数据访问速度
- **工作记忆**：纯内存存储，零延迟访问
- **感知缓冲区**：预分配避免内存分配开销

### 并发控制
- **读写锁**：每层使用 `std::shared_mutex` 实现高并发读取
- **细粒度锁**：不同索引使用独立锁减少竞争

### 索引优化
- **HNSW**：近似最近邻搜索，O(log n) 复杂度
- **RTree**：R* 树空间索引，高效范围查询
- **有序时间索引**：基于 `std::map` 的时间范围查询

---

## 5. 下一步优化建议

1. **分布式存储**：将长期记忆分片到多个节点
2. **增量学习**：实现更复杂的在线学习算法
3. **预测预加载**：基于访问模式预测并预加载记忆
4. **记忆压缩**：对长期记忆进行语义压缩
5. **跨模态融合**：增强多模态感知记忆的融合能力

---

## 6. 参考

- Atkinson, R.C. & Shiffrin, R.M. (1968). Human memory: A proposed system and its control processes.
- Baddeley, A.D. (2000). The episodic buffer: a new component of working memory?
- Miller, G.A. (1956). The magical number seven, plus or minus two.
- Malkov, Y.A. & Yashunin, D.A. (2020). Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs.

---

**优化日期**: 2026-02-10  
**版本**: v2.0
