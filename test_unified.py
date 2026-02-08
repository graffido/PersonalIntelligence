#!/usr/bin/env python3
"""
Unified Test Script for Personal Intelligence System
Tests all modules after merge to ensure imports work correctly.
"""

import sys
import os
import traceback
from pathlib import Path

# Add ml-service to path
sys.path.insert(0, str(Path(__file__).parent / "ml-service"))

# Test results
test_results = {
    "passed": [],
    "failed": [],
    "warnings": []
}

def test_import(module_name, description=""):
    """Test importing a module."""
    try:
        __import__(module_name)
        test_results["passed"].append(f"✓ {module_name}: {description or 'Import successful'}")
        return True
    except Exception as e:
        test_results["failed"].append(f"✗ {module_name}: {description or 'Import failed'} - {str(e)}")
        return False

def test_ml_service_modules():
    """Test ML service module imports."""
    print("\n" + "="*60)
    print("Testing ML Service Modules")
    print("="*60)
    
    modules = [
        ("enhanced_ner", "Enhanced NER with spaCy and jieba"),
        ("embedding_service", "Sentence embedding service"),
        ("llm_service", "LLM integration (OpenAI, Anthropic, Ollama)"),
        ("main", "FastAPI main application"),
        ("prediction_models", "Prediction and recommendation models"),
        ("recommendation_v2", "Recommendation engine v2"),
        ("advanced_reasoning", "Advanced reasoning capabilities"),
        ("privacy", "Privacy and data protection"),
        ("resilience", "Error handling and resilience"),
        ("data_importers", "Data import utilities"),
        ("logging_config", "Logging configuration"),
        ("vector_store_sqlite", "SQLite vector store"),
        ("client_example", "API client example"),
    ]
    
    for module, desc in modules:
        test_import(module, desc)

def test_config():
    """Test configuration module."""
    print("\n" + "="*60)
    print("Testing Configuration")
    print("="*60)
    
    try:
        import yaml
        config_path = Path(__file__).parent / "config" / "config.yaml"
        if config_path.exists():
            with open(config_path) as f:
                config = yaml.safe_load(f)
            test_results["passed"].append("✓ config.yaml: Configuration file loads correctly")
        else:
            test_results["warnings"].append("⚠ config.yaml: Configuration file not found")
    except Exception as e:
        test_results["failed"].append(f"✗ config.yaml: Failed to load - {str(e)}")

def test_pos_cpp():
    """Test C++ backend files exist."""
    print("\n" + "="*60)
    print("Testing C++ Backend (pos-cpp)")
    print("="*60)
    
    cpp_files = [
        "pos-cpp/src/core/memory/memory_store.cpp",
        "pos-cpp/src/core/memory/memory_store.h",
        "pos-cpp/src/core/ontology/ontology_graph.cpp",
        "pos-cpp/src/core/ontology/ontology_graph.h",
        "pos-cpp/src/core/common/types.h",
    ]
    
    for file in cpp_files:
        file_path = Path(__file__).parent / file
        if file_path.exists():
            test_results["passed"].append(f"✓ {file}: C++ file exists")
        else:
            test_results["warnings"].append(f"⚠ {file}: C++ file not found")

def test_pos_web():
    """Test React frontend files exist."""
    print("\n" + "="*60)
    print("Testing React Frontend (pos-web)")
    print("="*60)
    
    web_files = [
        "pos-web/package.json",
        "pos-web/src/App.tsx",
        "pos-web/src/main.tsx",
        "pos-web/src/components/UnifiedInput.tsx",
        "pos-web/src/components/KnowledgeGraphView.tsx",
        "pos-web/vite.config.ts",
    ]
    
    for file in web_files:
        file_path = Path(__file__).parent / file
        if file_path.exists():
            test_results["passed"].append(f"✓ {file}: Frontend file exists")
        else:
            test_results["failed"].append(f"✗ {file}: Frontend file not found")

def print_summary():
    """Print test summary."""
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = len(test_results["passed"])
    failed = len(test_results["failed"])
    warnings = len(test_results["warnings"])
    total = passed + failed + warnings
    
    print(f"\nTotal Tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Warnings: {warnings}")
    
    if test_results["failed"]:
        print("\n--- Failed Tests ---")
        for item in test_results["failed"]:
            print(f"  {item}")
    
    if test_results["warnings"]:
        print("\n--- Warnings ---")
        for item in test_results["warnings"]:
            print(f"  {item}")
    
    if test_results["passed"]:
        print(f"\n--- Passed Tests ({passed}) ---")
        for item in test_results["passed"][:10]:  # Show first 10
            print(f"  {item}")
        if passed > 10:
            print(f"  ... and {passed - 10} more")
    
    return failed == 0

def main():
    """Run all tests."""
    print("="*60)
    print("PERSONAL INTELLIGENCE SYSTEM - UNIFIED TEST SUITE")
    print("="*60)
    print(f"Working directory: {os.getcwd()}")
    
    try:
        test_ml_service_modules()
    except Exception as e:
        test_results["failed"].append(f"ML Service tests failed: {str(e)}")
        traceback.print_exc()
    
    try:
        test_config()
    except Exception as e:
        test_results["failed"].append(f"Config tests failed: {str(e)}")
        traceback.print_exc()
    
    try:
        test_pos_cpp()
    except Exception as e:
        test_results["failed"].append(f"C++ tests failed: {str(e)}")
        traceback.print_exc()
    
    try:
        test_pos_web()
    except Exception as e:
        test_results["failed"].append(f"Web tests failed: {str(e)}")
        traceback.print_exc()
    
    success = print_summary()
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
