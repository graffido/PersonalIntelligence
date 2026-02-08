#!/usr/bin/env python3
"""
ML服务客户端示例
演示如何使用各个API
"""
import json
import requests
from typing import Dict, List, Union


class MLServiceClient:
    """ML服务客户端"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")
    
    # ========== NER方法 ==========
    
    def extract_entities(self, text: str, extract_relations: bool = True) -> Dict:
        """抽取命名实体"""
        response = requests.post(
            f"{self.base_url}/ner/extract",
            json={"text": text, "extract_relations": extract_relations}
        )
        response.raise_for_status()
        return response.json()
    
    def add_to_ontology(self, text: str, entity_type: str, metadata: Dict = None) -> Dict:
        """添加实体到Ontology"""
        response = requests.post(
            f"{self.base_url}/ner/ontology/entity",
            json={"text": text, "entity_type": entity_type, "metadata": metadata}
        )
        response.raise_for_status()
        return response.json()
    
    def get_ontology_stats(self) -> Dict:
        """获取Ontology统计"""
        response = requests.get(f"{self.base_url}/ner/ontology/stats")
        response.raise_for_status()
        return response.json()
    
    # ========== Embedding方法 ==========
    
    def encode(self, texts: Union[str, List[str]], model: str = None) -> Dict:
        """编码文本"""
        response = requests.post(
            f"{self.base_url}/embedding/encode",
            json={"texts": texts, "model": model}
        )
        response.raise_for_status()
        return response.json()
    
    def similarity(self, text1: str, text2: str, model: str = None) -> float:
        """计算相似度"""
        response = requests.post(
            f"{self.base_url}/embedding/similarity",
            json={"text1": text1, "text2": text2, "model": model}
        )
        response.raise_for_status()
        return response.json()["similarity"]
    
    def search(self, query: str, documents: List[Dict], top_k: int = 5) -> List[Dict]:
        """语义搜索"""
        response = requests.post(
            f"{self.base_url}/embedding/search",
            json={"query": query, "documents": documents, "top_k": top_k}
        )
        response.raise_for_status()
        return response.json()["results"]
    
    # ========== LLM方法 ==========
    
    def chat(self, message: str, provider: str = None, stream: bool = False) -> str:
        """聊天"""
        if stream:
            response = requests.post(
                f"{self.base_url}/llm/chat",
                json={"messages": message, "provider": provider, "stream": True},
                stream=True
            )
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        data = line[6:]
                        if data == '[DONE]':
                            break
                        try:
                            chunk = json.loads(data)
                            print(chunk.get('content', ''), end='', flush=True)
                        except:
                            pass
            print()
            return ""
        else:
            response = requests.post(
                f"{self.base_url}/llm/chat/simple",
                params={"message": message, "provider": provider}
            )
            response.raise_for_status()
            return response.json()["content"]
    
    def health(self) -> Dict:
        """健康检查"""
        response = requests.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()


# ========== 示例用法 ==========

def demo_ner(client: MLServiceClient):
    """NER演示"""
    print("\n" + "="*50)
    print("🔍 NER实体抽取演示")
    print("="*50)
    
    # 中文示例
    chinese_text = "马化腾是腾讯公司的创始人，公司总部位于深圳。"
    print(f"\n文本: {chinese_text}")
    result = client.extract_entities(chinese_text)
    print(f"语言: {result['language']}")
    print(f"实体:")
    for entity in result['entities']:
        print(f"  - {entity['text']} ({entity['type']}) [{entity['confidence']:.2f}]")
    print(f"关系:")
    for relation in result['relations']:
        print(f"  - {relation['subject']['text']} --{relation['predicate']}--> {relation['object']['text']}")
    
    # 英文示例
    english_text = "Elon Musk founded SpaceX in California. The company is located in Hawthorne."
    print(f"\nText: {english_text}")
    result = client.extract_entities(english_text)
    print(f"Language: {result['language']}")
    print(f"Entities:")
    for entity in result['entities']:
        print(f"  - {entity['text']} ({entity['type']}) [{entity['confidence']:.2f}]")


def demo_embedding(client: MLServiceClient):
    """Embedding演示"""
    print("\n" + "="*50)
    print("📊 Embedding语义分析演示")
    print("="*50)
    
    # 相似度计算
    print("\n相似度计算:")
    pairs = [
        ("机器学习", "深度学习"),
        ("机器学习", "苹果香蕉"),
        ("自然语言处理", "NLP"),
    ]
    for t1, t2 in pairs:
        sim = client.similarity(t1, t2)
        print(f"  '{t1}' vs '{t2}': {sim:.4f}")
    
    # 语义搜索
    print("\n语义搜索:")
    docs = [
        {"id": 1, "title": "Python教程", "text": "Python是一种流行的编程语言"},
        {"id": 2, "title": "JavaScript", "text": "JavaScript用于网页开发"},
        {"id": 3, "title": "机器学习", "text": "机器学习是人工智能的一个分支"},
        {"id": 4, "title": "深度学习", "text": "深度学习使用神经网络进行学习"},
    ]
    results = client.search("AI技术", docs, top_k=2)
    for r in results:
        print(f"  {r['rank']}. {r['title']}: {r['text']} (相似度: {r['similarity']:.4f})")


def demo_llm(client: MLServiceClient):
    """LLM演示"""
    print("\n" + "="*50)
    print("🤖 LLM聊天演示")
    print("="*50)
    
    # 简单对话
    print("\n简单对话:")
    try:
        response = client.chat("你好，请用一句话介绍自己", provider="ollama")
        print(f"助手: {response}")
    except Exception as e:
        print(f"错误: {e}")
    
    # 流式输出
    print("\n流式输出 (使用OpenAI):")
    try:
        client.chat("写一首关于AI的短诗", provider="openai", stream=True)
    except Exception as e:
        print(f"错误: {e}")


def demo_ontology(client: MLServiceClient):
    """Ontology演示"""
    print("\n" + "="*50)
    print("🧠 Ontology知识图谱演示")
    print("="*50)
    
    # 添加实体
    print("\n添加实体到Ontology:")
    entities = [
        ("OpenAI", "ORGANIZATION", {"founded": 2015}),
        ("GPT-4", "PRODUCT", {"type": "LLM"}),
        ("Sam Altman", "PERSON", {"role": "CEO"}),
    ]
    for text, etype, meta in entities:
        result = client.add_to_ontology(text, etype, meta)
        print(f"  ✓ {text} ({etype})")
    
    # 获取统计
    print("\nOntology统计:")
    stats = client.get_ontology_stats()
    print(f"  总实体数: {stats.get('total_entities', 0)}")
    print(f"  总关系数: {stats.get('total_relations', 0)}")
    print(f"  实体类型分布: {stats.get('entities_by_type', {})}")


def main():
    """主函数"""
    client = MLServiceClient()
    
    # 检查服务状态
    print("检查服务状态...")
    try:
        health = client.health()
        print(f"✅ 服务状态: {health['status']}")
        print(f"服务可用性:")
        for service, available in health['services'].items():
            status = "✅" if available else "❌"
            print(f"  {status} {service}")
    except Exception as e:
        print(f"❌ 无法连接到服务: {e}")
        print("请确保服务已启动: python main.py")
        return
    
    # 运行演示
    demo_ner(client)
    demo_embedding(client)
    demo_ontology(client)
    demo_llm(client)
    
    print("\n" + "="*50)
    print("✨ 所有演示完成!")
    print("="*50)


if __name__ == "__main__":
    from typing import Union
    main()
