#!/bin/bash
# POS v3.0 一键启动脚本

set -e

POS_DIR="/Users/bingo/projects/personal_intelligence"

echo "======================================"
echo "POS v3.0 开发环境启动"
echo "======================================"

# 启动ML服务
echo "[1/4] 启动增强ML服务..."
cd "$POS_DIR/ml-service"
python3 enhanced_ml_service.py --port 8000 &
ML_PID=$!
echo "ML服务 PID: $ML_PID"

# 启动C++后端 (如果已编译)
echo "[2/4] 启动C++后端..."
if [ -f "$POS_DIR/pos-cpp/build/pos_server" ]; then
    cd "$POS_DIR/pos-cpp/build"
    ./pos_server --data-dir ./data --port 9000 &
    CORE_PID=$!
    echo "C++后端 PID: $CORE_PID"
else
    echo "C++后端未编译，使用Python版本..."
    cd "$POS_DIR"
    python3 pos_core_backend.py &
    CORE_PID=$!
    echo "Python后端 PID: $CORE_PID"
fi

# 启动前端
echo "[3/4] 启动前端..."
cd "$POS_DIR"
python3 -m http.server 8080 &
WEB_PID=$!
echo "Web服务 PID: $WEB_PID"

echo ""
echo "======================================"
echo "服务状态:"
echo "  ML服务:   http://localhost:8000"
echo "  后端API:  http://localhost:9000"
echo "  前端界面: http://localhost:8080/pos_core_frontend.html"
echo "======================================"
echo ""
echo "按 Ctrl+C 停止所有服务"
echo ""

# 保存PID
mkdir -p /tmp/pos
echo $ML_PID > /tmp/pos/ml.pid
echo $CORE_PID > /tmp/pos/core.pid
echo $WEB_PID > /tmp/pos/web.pid

# 等待
wait
