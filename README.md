# README - Personal Ontology System

## 系统架构

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Web Client    │────▶│   C++ Backend    │────▶│   ML Service    │
│   (React)       │     │   (REST API)     │     │   (Python)      │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                               │
        ┌──────────────────────┼──────────────────────┐
        ▼                      ▼                      ▼
┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│  Graph Store │      │ Vector Store │      │ Temporal/    │
│  (RocksDB)   │      │  (HNSW)      │      │ Spatial Store│
└──────────────┘      └──────────────┘      └──────────────┘
```

## 快速开始

### 1. 启动ML服务

```bash
cd ml-service
pip install -r requirements.txt
python main.py
```

服务将在 http://localhost:8000 启动

### 2. 构建C++后端

```bash
cd pos-cpp
mkdir build && cd build
cmake ..
make -j8
./pos_server
```

服务将在 http://localhost:8080 启动

### 3. 启动Web客户端

```bash
cd pos-web
npm install
npm run dev
```

访问 http://localhost:5173

## 功能特性

- ✅ 批量数据导入 (JSON/CSV)
- ✅ 手动数据录入 (模拟感知)
- ✅ 语义查询 (向量相似度)
- ✅ 时间查询 (范围查询)
- ✅ 概念查询 (本体引导)
- ✅ 时空联合查询 + 地图可视化
- ✅ 知识图谱可视化
- ✅ 实时统计面板

## API端点

| 端点 | 方法 | 描述 |
|------|------|------|
| `/health` | GET | 健康检查 |
| `/api/v1/ingest` | POST | 单条数据摄入 |
| `/api/v1/batch-import` | POST | 批量导入 |
| `/api/v1/query` | POST | 知识查询 |
| `/api/v1/spatiotemporal` | POST | 时空查询 |
| `/api/v1/graph` | POST | 图谱操作 |
| `/api/v1/stats` | GET | 统计信息 |
