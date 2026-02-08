# POS v3.0 开发路线图

## 当前状态
- ✅ POS Core Python原型 (运行中)
- 🔄 C++后端完整实现 (开发中)
- 🔄 增强ML服务 (开发中)
- 🔄 向量检索优化 (开发中)
- 🔄 现代化前端 (开发中)

## 模块分工

### 模块1: C++后端核心 (cpp-backend-complete)
负责：完整的C++后端编译和实现
- 存储层 (RocksDB + HNSW + RTree)
- 推理引擎
- HTTP API

### 模块2: 增强ML服务 (ml-service-enhanced)  
负责：智能实体抽取和embedding
- BERT/CRF NER
- Ontology-aware抽取
- 关系抽取

### 模块3: 功能与推理 (features-reasoning)
负责：数据导入和高级推理
- 日历/邮件/社交媒体导入
- 因果推理
- sqlite-vec向量检索

### 模块4: UX与健壮性 (ux-resilience)
负责：用户体验和系统稳定
- React前端v2
- 错误处理
- 配置管理

## 集成计划

```
Week 1: 各模块独立开发
Week 2: 模块间集成测试  
Week 3: 端到端测试和优化
Week 4: 部署和文档
```

## 关键技术选型

| 组件 | 技术 | 说明 |
|------|------|------|
| C++后端 | RocksDB + HNSW + Crow | 高性能存储和API |
| 向量检索 | sqlite-vec | SQLite向量扩展 |
| NER | BERT-CRF / spaCy | 深度学习实体抽取 |
| 推理 | 规则 + GNN | 混合推理 |
| 前端 | React + Tailwind + D3 | 现代化UI |
| 部署 | Docker Compose | 容器化 |

## 检查清单

- [ ] C++后端完整编译通过
- [ ] ML服务支持BERT NER
- [ ] sqlite-vec集成
- [ ] React前端组件化
- [ ] Docker化部署
- [ ] 端到端测试
