#include <iostream>
#include <filesystem>
#include <signal>
#include <yaml-cpp/yaml.h>

#include "core/ontology/ontology_graph.h"
#include "core/memory/memory_store.h"
#include "core/temporal/spatiotemporal_engine.h"
#include "ml/ml_service_client.h"
#include "api/http_server.h"

namespace pos {

// 全局指针用于信号处理
HttpServer* g_server = nullptr;

void signalHandler(int sig) {
    std::cout << "\nReceived signal " << sig << ", shutting down..." << std::endl;
    if (g_server) {
        g_server->stop();
    }
}

struct SystemConfig {
    AppConfig app;
    MLServiceConfig ml;
    std::string data_dir;
};

SystemConfig loadConfig(const std::string& config_path) {
    SystemConfig config;
    
    try {
        YAML::Node root = YAML::LoadFile(config_path);
        
        // 服务器配置
        if (root["server"]) {
            auto server = root["server"];
            config.app.host = server["host"].as<std::string>("0.0.0.0");
            config.app.port = server["port"].as<int>(8080);
            config.app.workers = server["workers"].as<int>(4);
            if (server["cors_origins"]) {
                for (const auto& origin : server["cors_origins"]) {
                    config.app.cors_origins.push_back(origin.as<std::string>());
                }
            }
        }
        
        // 数据目录
        config.data_dir = root["system"]["data_dir"].as<std::string>("./data");
        config.app.data_dir = config.data_dir;
        
        // ML服务配置
        if (root["ml_service"]) {
            auto ml = root["ml_service"];
            config.ml.endpoint = ml["endpoint"].as<std::string>("http://localhost:8000");
            config.ml.timeout_ms = ml["timeout_ms"].as<int>(5000);
            
            if (ml["models"]["embedding"]) {
                config.ml.embedding.name = ml["models"]["embedding"]["name"].as<std::string>();
                config.ml.embedding.dimension = ml["models"]["embedding"]["dimension"].as<int>(1536);
            }
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Warning: Failed to load config file: " << e.what() << std::endl;
        std::cerr << "Using default configuration" << std::endl;
    }
    
    return config;
}

} // namespace pos

int main(int argc, char* argv[]) {
    using namespace pos;
    
    std::cout << R"(
╔══════════════════════════════════════════════════════════════╗
║         Personal Ontology System (POS) v1.0.0                ║
║         个人本体记忆系统                                       ║
╚══════════════════════════════════════════════════════════════╝
)" << std::endl;
    
    // 解析命令行参数
    std::string config_path = "config/config.yaml";
    if (argc > 1) {
        config_path = argv[1];
    }
    
    // 加载配置
    auto config = loadConfig(config_path);
    
    // 创建数据目录
    std::filesystem::create_directories(config.data_dir);
    std::filesystem::create_directories(config.data_dir + "/memories");
    std::filesystem::create_directories(config.data_dir + "/ontology");
    std::filesystem::create_directories(config.data_dir + "/temporal");
    
    std::cout << "[Config] Data directory: " << config.data_dir << std::endl;
    std::cout << "[Config] Server: " << config.app.host << ":" << config.app.port << std::endl;
    std::cout << "[Config] ML Service: " << config.ml.endpoint << std::endl;
    
    try {
        // 初始化核心组件
        std::cout << "[Init] Loading ontology graph..." << std::endl;
        OntologyGraph ontology(config.data_dir + "/ontology");
        
        std::cout << "[Init] Loading memory store..." << std::endl;
        HierarchicalMemoryStore memory(config.data_dir + "/memories", config.ml.embedding.dimension);
        
        std::cout << "[Init] Loading spatiotemporal index..." << std::endl;
        SpatiotemporalIndex st_index(config.data_dir + "/temporal");
        
        std::cout << "[Init] Connecting to ML service..." << std::endl;
        MLServiceClient ml_client(config.ml);
        
        // 检查ML服务健康
        if (ml_client.healthCheck()) {
            std::cout << "[Init] ML service connected ✓" << std::endl;
        } else {
            std::cout << "[Warning] ML service not available, some features will be limited" << std::endl;
        }
        
        // 创建HTTP服务器
        HttpServer server(config.app, ontology, memory, st_index, ml_client);
        g_server = &server;
        
        // 注册信号处理
        signal(SIGINT, signalHandler);
        signal(SIGTERM, signalHandler);
        
        // 启动服务器
        std::cout << "[Init] Starting HTTP server..." << std::endl;
        if (server.start()) {
            std::cout << "[Ready] Server running at http://" << config.app.host << ":" << config.app.port << std::endl;
            std::cout << "[Ready] Press Ctrl+C to stop" << std::endl;
            
            // 保持运行直到收到信号
            while (server.isRunning()) {
                std::this_thread::sleep_for(std::chrono::seconds(1));
            }
        } else {
            std::cerr << "[Error] Failed to start server" << std::endl;
            return 1;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "[Fatal Error] " << e.what() << std::endl;
        return 1;
    }
    
    std::cout << "[Shutdown] Goodbye!" << std::endl;
    return 0;
}
