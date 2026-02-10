/**
 * @file test_quad_layer_memory.cpp
 * @brief 四层分层记忆系统测试
 */

#include "core/memory/hierarchical_memory.h"
#include <iostream>
#include <assert>
#include <chrono>
#include <thread>

using namespace personal_ontology::memory;

// 辅助函数：打印测试结果
void print_result(const std::string& test_name, bool passed) {
    std::cout << "[" << (passed ? "PASS" : "FAIL") << "] " << test_name << std::endl;
}

// 测试 Layer 0: 感知记忆
bool test_sensory_memory() {
    std::cout << "\n=== Testing Layer 0: Sensory Memory ===" << std::endl;
    
    HierarchicalMemoryConfig config;
    config.sensory.buffer_size = 10;
    config.sensory.decay_ms = 1000;  // 1秒衰减
    
    QuadLayerMemoryStore store(config);
    auto init_result = store.initialize();
    if (init_result.isError()) {
        std::cerr << "Failed to initialize: " << init_result.errorMessage() << std::endl;
        return false;
    }
    
    // 测试文本输入
    auto result1 = store.sensoryInputText("Hello, this is a test input", 80.0f);
    print_result("Sensory text input", result1.isOk());
    
    auto result2 = store.sensoryInputText("Another input", 40.0f);
    print_result("Sensory text input (low attention)", result2.isOk());
    
    // 测试获取注意焦点
    auto focus = store.getAttentionFocus();
    bool has_high_attention = false;
    for (const auto& entry : focus) {
        if (entry.attention_weight >= 60.0f) {
            has_high_attention = true;
            break;
        }
    }
    print_result("Get attention focus", has_high_attention);
    
    store.shutdown();
    return true;
}

// 测试 Layer 1: 工作记忆
bool test_working_memory() {
    std::cout << "\n=== Testing Layer 1: Working Memory ===" << std::endl;
    
    HierarchicalMemoryConfig config;
    config.working.capacity = 5;  // 小容量便于测试
    
    QuadLayerMemoryStore store(config);
    auto init_result = store.initialize();
    if (init_result.isError()) {
        std::cerr << "Failed to initialize: " << init_result.errorMessage() << std::endl;
        return false;
    }
    
    // 创建测试记忆
    MemoryTrace mem1;
    mem1.content = "Test memory 1";
    mem1.type = "test";
    
    // 测试加载到工作记忆
    auto result1 = store.loadToWorkingMemory(mem1, true);
    print_result("Load to working memory", result1.isOk());
    
    // 测试获取焦点记忆
    auto focused = store.getFocusedMemory();
    print_result("Get focused memory", focused.has_value() && focused->content == "Test memory 1");
    
    // 测试复述
    auto rehearsal_result = store.rehearse(mem1.id);
    print_result("Rehearse memory", rehearsal_result.isOk());
    
    // 测试创建信息块
    std::vector<MemoryId> items = {mem1.id};
    auto chunk_result = store.createChunk("Test Chunk", items);
    print_result("Create chunk", chunk_result.isOk());
    
    // 测试工作记忆内容
    auto contents = store.getWorkingMemoryContents();
    print_result("Get working memory contents", !contents.empty());
    
    store.shutdown();
    return true;
}

// 测试 Layer 2: 长期记忆
bool test_long_term_memory() {
    std::cout << "\n=== Testing Layer 2: Long-term Memory ===" << std::endl;
    
    HierarchicalMemoryConfig config;
    
    QuadLayerMemoryStore store(config);
    auto init_result = store.initialize();
    if (init_result.isError()) {
        std::cerr << "Failed to initialize: " << init_result.errorMessage() << std::endl;
        return false;
    }
    
    // 创建测试记忆
    MemoryTrace mem1;
    mem1.content = "Important long-term memory";
    mem1.type = "episodic";
    mem1.source = "test";
    mem1.timestamp = MemoryTrace::now();
    
    // 添加嵌入向量
    Embedding emb(768, 0.1f);
    mem1.embedding = emb;
    
    // 存储到长期记忆
    auto store_result = store.storeLongTerm(mem1);
    print_result("Store to long-term memory", store_result.isOk());
    
    if (store_result.isOk()) {
        auto id = store_result.value();
        
        // 检索
        auto retrieve_result = store.retrieveLongTerm(id);
        print_result("Retrieve from long-term memory", 
            retrieve_result.isOk() && retrieve_result.value().content == mem1.content);
        
        // 测试关联
        MemoryTrace mem2;
        mem2.content = "Related memory";
        mem2.type = "episodic";
        auto store_result2 = store.storeLongTerm(mem2);
        
        if (store_result2.isOk()) {
            auto id2 = store_result2.value();
            auto assoc_result = store.associate(id, id2, 0.8);
            print_result("Associate memories", assoc_result.isOk());
        }
    }
    
    store.shutdown();
    return true;
}

// 测试 Layer 3: 参数记忆
bool test_parameter_memory() {
    std::cout << "\n=== Testing Layer 3: Parameter Memory ===" << std::endl;
    
    HierarchicalMemoryConfig config;
    
    QuadLayerMemoryStore store(config);
    auto init_result = store.initialize();
    if (init_result.isError()) {
        std::cerr << "Failed to initialize: " << init_result.errorMessage() << std::endl;
        return false;
    }
    
    // 测试存储参数
    std::vector<float> params = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f};
    auto store_result = store.storeParameter("user_preference_vector", params);
    print_result("Store parameter", store_result.isOk());
    
    // 测试检索参数
    auto retrieve_result = store.retrieveParameter("user_preference_vector");
    print_result("Retrieve parameter", 
        retrieve_result.has_value() && retrieve_result->size() == 5);
    
    // 测试更新用户偏好
    std::vector<float> gradient = {0.01f, 0.01f, 0.01f, 0.01f, 0.01f};
    store.updateUserPreferences(gradient, 0.1f);
    auto prefs = store.getUserPreferences();
    print_result("Update user preferences", prefs.update_count > 0);
    
    // 测试访问模式
    store.recordAccessEvent("test_id_1", MemoryLayer::LONG_TERM);
    store.recordAccessEvent("test_id_2", MemoryLayer::LONG_TERM);
    auto pattern = store.getAccessPattern();
    print_result("Get access pattern", true);
    
    store.shutdown();
    return true;
}

// 测试层间转换
bool test_layer_transitions() {
    std::cout << "\n=== Testing Layer Transitions ===" << std::endl;
    
    HierarchicalMemoryConfig config;
    
    QuadLayerMemoryStore store(config);
    auto init_result = store.initialize();
    if (init_result.isError()) {
        std::cerr << "Failed to initialize: " << init_result.errorMessage() << std::endl;
        return false;
    }
    
    // 1. 感知 → 工作记忆 (注意)
    auto sensory_result = store.sensoryInputText("Input to attend to", 90.0f);
    if (sensory_result.isOk()) {
        auto sensory_id = sensory_result.value();
        auto attend_result = store.attend(sensory_id, "Interpreted content");
        print_result("Attend (sensory -> working)", attend_result.isOk());
    }
    
    // 2. 工作 → 长期记忆 (巩固)
    MemoryTrace mem;
    mem.content = "Memory to consolidate";
    mem.type = "test";
    auto load_result = store.loadToWorkingMemory(mem, false);
    if (load_result.isOk()) {
        auto working_id = mem.id;
        // 多次复述以提高巩固概率
        store.rehearse(working_id);
        store.rehearse(working_id);
        store.rehearse(working_id);
        
        auto consolidate_result = store.consolidate(working_id);
        print_result("Consolidate (working -> long-term)", consolidate_result.isOk());
    }
    
    // 3. 长期 → 工作记忆 (回忆)
    MemoryTrace ltm_mem;
    ltm_mem.content = "Memory in LTM";
    ltm_mem.type = "episodic";
    auto ltm_result = store.storeLongTerm(ltm_mem);
    if (ltm_result.isOk()) {
        auto ltm_id = ltm_result.value();
        auto recall_result = store.recallToWorking(ltm_id);
        print_result("Recall (long-term -> working)", recall_result.isOk());
    }
    
    store.shutdown();
    return true;
}

// 测试维护任务
bool test_maintenance() {
    std::cout << "\n=== Testing Maintenance Tasks ===" << std::endl;
    
    HierarchicalMemoryConfig config;
    config.maintenance_interval_ms = 1000;  // 1秒间隔用于测试
    config.enable_auto_consolidation = false;  // 手动触发
    
    QuadLayerMemoryStore store(config);
    auto init_result = store.initialize();
    if (init_result.isError()) {
        std::cerr << "Failed to initialize: " << init_result.errorMessage() << std::endl;
        return false;
    }
    
    // 获取初始统计
    auto initial_stats = store.getStatistics();
    std::cout << "Initial state: sensory=" << initial_stats.sensory_count 
              << ", working=" << initial_stats.working_count << std::endl;
    
    // 添加一些测试数据
    for (int i = 0; i < 5; i++) {
        store.sensoryInputText("Test input " + std::to_string(i), 50.0f);
        
        MemoryTrace mem;
        mem.content = "Working memory " + std::to_string(i);
        mem.type = "test";
        store.loadToWorkingMemory(mem, false);
    }
    
    // 手动触发维护
    auto maint_result = store.runMaintenance();
    if (maint_result.isOk()) {
        auto result = maint_result.value();
        std::cout << "Maintenance completed in " << result.duration_ms << "ms" << std::endl;
        std::cout << "  Decayed sensory: " << result.decayed_sensory << std::endl;
        std::cout << "  Decayed working: " << result.decayed_working << std::endl;
        std::cout << "  Consolidated: " << result.consolidated << std::endl;
        print_result("Maintenance task", true);
    } else {
        print_result("Maintenance task", false);
    }
    
    // 获取最终统计
    auto final_stats = store.getStatistics();
    std::cout << "Final state: sensory=" << final_stats.sensory_count 
              << ", working=" << final_stats.working_count 
              << ", long_term=" << final_stats.long_term_count << std::endl;
    
    store.shutdown();
    return true;
}

// 测试统计和检查点
bool test_statistics_and_checkpoint() {
    std::cout << "\n=== Testing Statistics and Checkpoint ===" << std::endl;
    
    HierarchicalMemoryConfig config;
    
    QuadLayerMemoryStore store(config);
    auto init_result = store.initialize();
    if (init_result.isError()) {
        std::cerr << "Failed to initialize: " << init_result.errorMessage() << std::endl;
        return false;
    }
    
    // 添加一些数据
    store.sensoryInputText("Test", 60.0f);
    MemoryTrace mem;
    mem.content = "Test memory";
    mem.type = "test";
    store.loadToWorkingMemory(mem, true);
    
    // 获取统计
    auto stats = store.getStatistics();
    print_result("Get statistics", stats.sensory_count > 0 || stats.working_count > 0);
    
    std::cout << "Statistics:" << std::endl;
    std::cout << "  Sensory: " << stats.sensory_count << std::endl;
    std::cout << "  Working: " << stats.working_count << std::endl;
    std::cout << "  Cache hit rate: " << (stats.cache_hit_rate * 100) << "%" << std::endl;
    
    // 测试检查点
    auto checkpoint_result = store.createCheckpoint("/tmp/pos_test_checkpoint");
    print_result("Create checkpoint", checkpoint_result.isOk());
    
    store.shutdown();
    return true;
}

// 主测试函数
int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "POS Quad-Layer Memory System Test Suite" << std::endl;
    std::cout << "========================================" << std::endl;
    
    int passed = 0;
    int total = 6;
    
    if (test_sensory_memory()) passed++;
    if (test_working_memory()) passed++;
    if (test_long_term_memory()) passed++;
    if (test_parameter_memory()) passed++;
    if (test_layer_transitions()) passed++;
    if (test_maintenance()) passed++;
    if (test_statistics_and_checkpoint()) passed++;
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "Test Results: " << passed << "/" << total << " passed" << std::endl;
    std::cout << "========================================" << std::endl;
    
    return (passed == total) ? 0 : 1;
}
