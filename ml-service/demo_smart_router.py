#!/usr/bin/env python3
"""
POS 智能查询路由演示
展示如何通过本体推理减少LLM调用
"""

import time
from datetime import datetime
from reasoning_engine import LocalReasoningEngine, Concept, Memory

class SmartQueryRouter:
    """智能查询路由器 - 优先使用本地推理"""
    
    def __init__(self):
        self.local_engine = LocalReasoningEngine()
        self.llm_call_count = 0
        self.local_inference_count = 0
    
    def add_knowledge(self, concept, memory):
        """添加知识到本地引擎"""
        self.local_engine.add_concept(concept)
        self.local_engine.add_memory(memory)
    
    def query(self, user_query: str, entities: list) -> dict:
        """
        处理用户查询
        
        流程:
        1. 尝试本地推理
        2. 评估本地结果质量
        3. 仅在必要时调用LLM
        """
        print(f"\n{'='*60}")
        print(f"用户查询: \"{user_query}\"")
        print(f"{'='*60}")
        
        # 步骤1: 本地推理
        print("\n[步骤1] 执行本地推理...")
        start_time = time.time()
        
        local_result = self.local_engine.query_with_reasoning(user_query, entities)
        inference_results = self.local_engine.infer()
        
        local_time = time.time() - start_time
        self.local_inference_count += 1
        
        print(f"  ✓ 本地推理完成 ({local_time:.3f}s)")
        print(f"  - 直接匹配: {len(local_result['direct_matches'])} 条")
        print(f"  - 推理匹配: {len(local_result['inferred_matches'])} 条")
        print(f"  - 推理规则触发: {len(inference_results)} 个")
        
        # 步骤2: 评估是否需要LLM
        print("\n[步骤2] 评估结果质量...")
        
        quality_score = self._evaluate_quality(local_result, inference_results)
        print(f"  本地推理质量评分: {quality_score:.2f}/1.0")
        
        # 步骤3: 决定路由
        if quality_score >= 0.7:
            print(f"\n[决策] ✅ 本地推理质量足够，无需调用LLM")
            return self._format_local_result(local_result, inference_results)
        else:
            print(f"\n[决策] ⚠️ 本地推理不足，需要调用LLM")
            return self._call_llm(user_query, local_result)
    
    def _evaluate_quality(self, local_result, inference_results) -> float:
        """评估本地推理结果质量"""
        score = 0.0
        
        # 有直接匹配加分
        if local_result['direct_matches']:
            score += 0.4
            if any(m.get('match_type') == 'direct' for m in local_result['direct_matches']):
                score += 0.2
        
        # 有推理匹配加分
        if local_result['inferred_matches']:
            score += 0.2
        
        # 有实用建议加分
        if local_result['suggestions']:
            score += 0.1
            if any(s.get('confidence', 0) > 0.7 for s in local_result['suggestions']):
                score += 0.1
        
        return min(score, 1.0)
    
    def _format_local_result(self, local_result, inference_results) -> dict:
        """格式化本地推理结果"""
        return {
            "source": "local_reasoning",
            "direct_matches": local_result['direct_matches'][:5],
            "inferred_matches": local_result['inferred_matches'][:5],
            "suggestions": local_result['suggestions'][:3],
            "reasoning_details": [
                {
                    "rule": r.rule_name,
                    "description": r.description,
                    "confidence": r.confidence
                }
                for r in inference_results[:3]
            ],
            "llm_called": False,
            "reasoning_path": [
                "1. 实体识别与本体匹配",
                "2. 关系图谱遍历",
                "3. 模式匹配",
                "4. 本地推理生成回复"
            ]
        }
    
    def _call_llm(self, user_query: str, local_context: dict) -> dict:
        """模拟调用LLM"""
        self.llm_call_count += 1
        
        print("  正在调用LLM...")
        time.sleep(0.5)  # 模拟延迟
        
        return {
            "source": "llm",
            "response": f"[LLM回复] 基于本地检索到的 {len(local_context['direct_matches'])} 条记忆，...",
            "llm_called": True,
            "local_context": local_context
        }
    
    def get_stats(self):
        """获取统计信息"""
        total = self.local_inference_count + self.llm_call_count
        llm_ratio = self.llm_call_count / total if total > 0 else 0
        
        return {
            "local_inferences": self.local_inference_count,
            "llm_calls": self.llm_call_count,
            "total_queries": total,
            "llm_usage_ratio": f"{llm_ratio*100:.1f}%",
            "cost_savings": f"{(1-llm_ratio)*100:.1f}%"
        }


def main():
    print("="*60)
    print("POS 智能查询路由演示")
    print("展示本体推理如何减少LLM调用")
    print("="*60)
    
    router = SmartQueryRouter()
    
    # 构建知识库
    print("\n[初始化] 构建本地知识库...")
    
    # 添加人物概念
    router.add_knowledge(
        Concept("person_zhangsan", "PERSON", "张三", relations=[]),
        Memory("mem_init", "初始化", datetime.now(), [])
    )
    router.add_knowledge(
        Concept("person_lisi", "PERSON", "李四", relations=[
            {"type": "KNOWS", "target": "person_zhangsan", "weight": 0.9}
        ]),
        Memory("mem_init", "初始化", datetime.now(), [])
    )
    router.add_knowledge(
        Concept("place_starbucks", "PLACE", "星巴克"),
        Memory("mem_init", "初始化", datetime.now(), [])
    )
    
    # 添加多条记忆（模拟历史数据）
    from datetime import timedelta
    
    memories = [
        ("今天早晨在星巴克和李四讨论项目方案，确定了技术选型", 
         [{"text": "李四", "label": "PERSON"}, {"text": "星巴克", "label": "PLACE"}]),
        ("下午在健身房遇到张三，一起锻炼了1小时",
         [{"text": "张三", "label": "PERSON"}, {"text": "健身房", "label": "PLACE"}]),
        ("晚上和女朋友在海底捞吃火锅庆祝纪念日",
         [{"text": "女朋友", "label": "PERSON"}, {"text": "海底捞", "label": "PLACE"}]),
        ("昨天在星巴克加班赶项目进度",
         [{"text": "星巴克", "label": "PLACE"}]),
        ("周三上午在公司参加全员会议",
         [{"text": "公司", "label": "PLACE"}]),
    ]
    
    for i, (content, entities) in enumerate(memories):
        router.add_knowledge(
            Concept(f"concept_{i}", "EVENT", f"event_{i}"),
            Memory(
                id=f"mem_{i}",
                content=content,
                timestamp=datetime.now() - timedelta(days=i),
                entities=entities
            )
        )
    
    print(f"  ✓ 已添加 {len(memories)} 条记忆")
    
    # 测试场景
    test_queries = [
        {
            "query": "找一下我和李四在星巴克的记忆",
            "entities": [{"text": "李四", "label": "PERSON"}, {"text": "星巴克", "label": "PLACE"}],
            "expected": "本地推理（有精确匹配）"
        },
        {
            "query": "最近有什么活动模式",
            "entities": [],
            "expected": "本地推理（模式发现）"
        },
        {
            "query": "为什么我总是晚上吃火锅？",
            "entities": [],
            "expected": "LLM（需要解释性推理）"
        },
        {
            "query": "张三和李四认识吗？",
            "entities": [{"text": "张三", "label": "PERSON"}, {"text": "李四", "label": "PERSON"}],
            "expected": "本地推理（关系推理）"
        },
    ]
    
    print("\n" + "="*60)
    print("开始测试查询场景")
    print("="*60)
    
    for i, test in enumerate(test_queries, 1):
        result = router.query(test["query"], test["entities"])
        
        print(f"\n结果来源: {'本地推理' if not result['llm_called'] else 'LLM'}")
        if result.get('suggestions'):
            print(f"建议: {result['suggestions'][0].get('description', 'N/A')[:50]}...")
    
    # 统计
    print("\n" + "="*60)
    print("统计汇总")
    print("="*60)
    stats = router.get_stats()
    print(f"本地推理次数: {stats['local_inferences']}")
    print(f"LLM调用次数: {stats['llm_calls']}")
    print(f"LLM使用率: {stats['llm_usage_ratio']}")
    print(f"成本节约: {stats['cost_savings']}")
    
    print("\n" + "="*60)
    print("演示完成!")
    print("="*60)


if __name__ == "__main__":
    main()
