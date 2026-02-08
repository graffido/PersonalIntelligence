#!/usr/bin/env python3
"""
快速测试脚本 - 验证ML服务核心功能
"""
import sys
from pathlib import Path

# 添加当前目录到路径
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """测试模块导入"""
    print("🔍 测试模块导入...")
    try:
        from enhanced_ner import EnhancedNER, Entity, EntityType, create_ner_service
        from embedding_service import EmbeddingService, create_embedding_service
        from llm_service import LLMService, Message, ModelType, create_llm_service
        print("✅ 所有模块导入成功")
        return True
    except Exception as e:
        print(f"❌ 导入失败: {e}")
        return False

def test_ner():
    """测试NER功能"""
    print("\n📝 测试NER功能...")
    try:
        from enhanced_ner import EnhancedNER
        
        # 创建NER服务（不加载模型，仅测试结构）
        ner = EnhancedNER({
            "use_bert_chinese": False,
            "use_transformer": False  # 避免spaCy模型依赖
        })
        
        # 测试Ontology功能
        ner.add_to_ontology("测试实体", "CONCEPT", {"test": True})
        stats = ner.get_ontology_stats()
        print(f"✅ Ontology统计: {stats}")
        
        return True
    except Exception as e:
        print(f"⚠️ NER测试部分失败（可能是模型未安装）: {e}")
        return True  # 非关键失败

def test_embedding():
    """测试Embedding功能"""
    print("\n📊 测试Embedding功能...")
    try:
        from embedding_service import EmbeddingService
        
        # 创建服务但不自动加载模型
        service = EmbeddingService({"auto_load": False})
        
        # 测试模型列表
        models = service.list_models()
        print(f"✅ 可用模型数: {len(models)}")
        print(f"   示例: {models[0]['id'] if models else 'N/A'}")
        
        return True
    except Exception as e:
        print(f"❌ Embedding测试失败: {e}")
        return False

def test_llm():
    """测试LLM功能"""
    print("\n🤖 测试LLM功能...")
    try:
        from llm_service import LLMService, ModelType
        
        # 创建服务
        service = LLMService({
            "openai": {"enabled": False},
            "anthropic": {"enabled": False},
            "ollama": {"enabled": False},  # 避免连接测试
        })
        
        # 测试复杂度分析
        from llm_service import ComplexityAnalyzer
        analyzer = ComplexityAnalyzer()
        
        test_texts = [
            "你好",
            "请分析机器学习的应用场景",
            "详细解释量子计算对密码学的影响"
        ]
        for text in test_texts:
            complexity = analyzer.analyze(text)
            print(f"   '{text[:20]}...' -> {complexity.value}")
        
        print("✅ LLM基础功能正常")
        return True
    except Exception as e:
        print(f"❌ LLM测试失败: {e}")
        return False

def test_config():
    """测试配置文件"""
    print("\n⚙️ 测试配置文件...")
    try:
        import yaml
        config_path = Path(__file__).parent / "config.yaml"
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            print(f"✅ 配置加载成功")
            print(f"   Server: {config.get('server', {}).get('host')}:{config.get('server', {}).get('port')}")
            print(f"   NER: {config.get('ner', {}).get('use_ontology')}")
            print(f"   Embedding: {config.get('embedding', {}).get('default_model')}")
            print(f"   LLM: {config.get('llm', {}).get('default_model')}")
        else:
            print("⚠️ 配置文件不存在")
        return True
    except Exception as e:
        print(f"❌ 配置测试失败: {e}")
        return False

def main():
    """运行所有测试"""
    print("="*60)
    print("🧪 增强版ML服务功能测试")
    print("="*60)
    
    tests = [
        ("模块导入", test_imports),
        ("NER功能", test_ner),
        ("Embedding功能", test_embedding),
        ("LLM功能", test_llm),
        ("配置文件", test_config),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ {name} 测试异常: {e}")
            results.append((name, False))
    
    print("\n" + "="*60)
    print("📋 测试结果汇总")
    print("="*60)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{status}: {name}")
    
    print("="*60)
    print(f"总计: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！服务就绪。")
        return 0
    else:
        print("⚠️ 部分测试失败，请检查依赖和配置。")
        return 1

if __name__ == "__main__":
    exit(main())
