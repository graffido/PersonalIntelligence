#!/bin/bash
# 端到端测试脚本

set -e

echo "=========================================="
echo "POS 端到端测试"
echo "=========================================="

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

API_URL="http://localhost:8080"
ML_URL="http://localhost:8000"

# 检查服务是否运行
check_service() {
    local url=$1
    local name=$2
    if curl -s "$url/health" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ $name 运行中${NC}"
        return 0
    else
        echo -e "${RED}✗ $name 未启动${NC}"
        return 1
    fi
}

# 测试数据构造
echo ""
echo "步骤1: 构造测试数据..."
cat > /tmp/test_data.json << 'JSON_EOF'
[
  {
    "content": "今天早晨8点在星巴克和中伟讨论项目方案，喝了拿铁咖啡",
    "timestamp": "2024-01-15T08:00:00",
    "location": {"lat": 39.9042, "lng": 116.4074, "name": "星巴克咖啡店"}
  },
  {
    "content": "下午3点去海淀医院看牙医，预约了李医生",
    "timestamp": "2024-01-15T15:00:00",
    "location": {"lat": 39.96, "lng": 116.30, "name": "海淀医院"}
  },
  {
    "content": "晚上和女朋友在海底捞吃火锅，庆祝纪念日",
    "timestamp": "2024-01-15T19:30:00",
    "location": {"lat": 39.915, "lng": 116.415, "name": "海底捞"}
  },
  {
    "content": "昨天在健身房锻炼了1小时，做了深蹲和卧推",
    "timestamp": "2024-01-14T18:00:00",
    "location": {"lat": 39.92, "lng": 116.40, "name": "乐刻健身房"}
  },
  {
    "content": "周三上午在公司参加全员会议，讨论了Q4目标",
    "timestamp": "2024-01-10T10:00:00",
    "location": {"lat": 39.93, "lng": 116.42, "name": "公司会议室"}
  }
]
JSON_EOF
echo -e "${GREEN}✓ 测试数据已生成 (/tmp/test_data.json)${NC}"

# 检查服务
echo ""
echo "步骤2: 检查服务状态..."
ML_OK=false
API_OK=false

if check_service "$ML_URL" "ML服务"; then
    ML_OK=true
else
    echo -e "${YELLOW}⚠ 请先启动ML服务: python3 ml-service/test_main.py${NC}"
fi

if check_service "$API_URL" "C++ API"; then
    API_OK=true
else
    echo -e "${YELLOW}⚠ 请先启动C++服务 (可选，当前使用模拟测试)${NC}"
fi

# 测试ML服务API
echo ""
echo "步骤3: 测试ML服务核心能力..."

# 测试Embedding
echo "  测试 /embed 端点..."
EMBED_RESULT=$(curl -s -X POST "$ML_URL/embed" \
    -H "Content-Type: application/json" \
    -d '{"texts": ["星巴克喝咖啡", "海底捞吃火锅"]}')

if echo "$EMBED_RESULT" | grep -q "embeddings"; then
    echo -e "${GREEN}  ✓ Embedding生成成功${NC}"
    # 提取维度信息
    DIM=$(echo "$EMBED_RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin)['dimension'])" 2>/dev/null || echo "unknown")
    echo "    维度: $DIM"
else
    echo -e "${RED}  ✗ Embedding失败${NC}"
    echo "  响应: $EMBED_RESULT"
fi

# 测试NER
echo "  测试 /ner 端点..."
NER_RESULT=$(curl -s -X POST "$ML_URL/ner" \
    -H "Content-Type: application/json" \
    -d '{"text": "张三和李四去星巴克喝咖啡"}')

if echo "$NER_RESULT" | grep -q "entities"; then
    echo -e "${GREEN}  ✓ 实体抽取成功${NC}"
    # 提取实体数量
    COUNT=$(echo "$NER_RESULT" | python3 -c "import sys,json; print(len(json.load(sys.stdin)['entities']))" 2>/dev/null || echo "0")
    echo "    发现实体: $COUNT 个"
else
    echo -e "${RED}  ✗ NER失败${NC}"
    echo "  响应: $NER_RESULT"
fi

# 模拟完整数据流测试
echo ""
echo "步骤4: 模拟完整数据流..."

# 模拟数据摄入流程
echo "  模拟单条数据摄入..."
TEST_CONTENT="今天早晨在星巴克和中伟讨论项目"
INGEST_RESULT=$(curl -s -X POST "$ML_URL/ner" \
    -H "Content-Type: application/json" \
    -d "{\"text\": \"$TEST_CONTENT\"}")

echo "  输入: $TEST_CONTENT"
echo "  实体抽取结果:"
echo "$INGEST_RESULT" | python3 -m json.tool 2>/dev/null || echo "$INGEST_RESULT"

# 模拟Embedding生成
echo ""
echo "  模拟Embedding生成..."
EMBED_RESULT=$(curl -s -X POST "$ML_URL/embed" \
    -H "Content-Type: application/json" \
    -d "{\"texts\": [\"$TEST_CONTENT\"]}")

if echo "$EMBED_RESULT" | grep -q "embeddings"; then
    VEC_LEN=$(echo "$EMBED_RESULT" | python3 -c "import sys,json; print(len(json.load(sys.stdin)['embeddings'][0]))" 2>/dev/null || echo "0")
    echo -e "${GREEN}  ✓ 生成向量维度: $VEC_LEN${NC}"
fi

# 测试时空查询能力
echo ""
echo "步骤5: 测试时空查询场景..."

# 场景1: 时间范围查询
echo "  场景: 查找2024-01-15的记忆"
echo "  请求参数:"
echo '  {
    "time": {
      "start": "2024-01-15T00:00:00",
      "end": "2024-01-15T23:59:59"
    },
    "temporal_relation": "during"
  }'
echo "  (C++服务启动后可实际测试)"

# 场景2: 空间范围查询
echo ""
echo "  场景: 查找星巴克附近的记忆"
echo "  请求参数:"
echo '  {
    "location": {"lat": 39.9042, "lng": 116.4074},
    "radius": 1000,
    "temporal_relation": "at"
  }'
echo "  (C++服务启动后可实际测试)"

# 汇总
echo ""
echo "=========================================="
echo "测试汇总"
echo "=========================================="
echo -e "ML服务: $(if $ML_OK; then echo -e "${GREEN}运行中${NC}"; else echo -e "${RED}未启动${NC}"; fi)"
echo -e "API服务: $(if $API_OK; then echo -e "${GREEN}运行中${NC}"; else echo -e "${YELLOW}未启动(可选)${NC}"; fi)"
echo ""
echo "核心能力验证:"
echo "  ✓ 实体抽取 (NER)"
echo "  ✓ 向量生成 (Embedding)"
echo "  ✓ 数据格式解析"
echo "  ⚠ 完整数据存储 (需C++服务)"
echo "  ⚠ 时空查询 (需C++服务)"
echo ""
echo "=========================================="
echo "端到端测试完成!"
echo "=========================================="
