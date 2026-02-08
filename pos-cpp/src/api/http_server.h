#pragma once

#include "core/ontology/ontology_graph.h"
#include "core/memory/memory_store.h"
#include "core/temporal/spatiotemporal_engine.h"
#include "ml/ml_service_client.h"
#include <crow.h>

namespace pos {

// 应用配置
struct AppConfig {
    std::string host{"0.0.0.0"};
    int port{8080};
    int workers{4};
    std::vector<std::string> cors_origins{"http://localhost:5173"};
    
    std::string data_dir{"./data"};
    std::string log_level{"info"};
};

// HTTP服务器
class HttpServer {
public:
    HttpServer(const AppConfig& config,
               OntologyGraph& ontology,
               HierarchicalMemoryStore& memory,
               SpatiotemporalIndex& st_index,
               MLServiceClient& ml_client);
    
    ~HttpServer();
    
    bool start();
    void stop();
    bool isRunning() const;
    
    // 获取实际端口 (如果绑定到0)
    int getPort() const;

private:
    class Impl;
    std::unique_ptr<Impl> pimpl_;
};

// 请求处理器
class RequestHandlers {
public:
    // 数据摄入
    static crow::response handleIngest(const crow::request& req,
                                      HierarchicalMemoryStore& memory,
                                      OntologyGraph& ontology,
                                      MLServiceClient& ml);
    
    // 批量导入
    static crow::response handleBatchImport(const crow::request& req,
                                           HierarchicalMemoryStore& memory,
                                           OntologyGraph& ontology,
                                           MLServiceClient& ml);
    
    // 知识查询
    static crow::response handleQuery(const crow::request& req,
                                     HierarchicalMemoryStore& memory,
                                     OntologyGraph& ontology);
    
    // 时空查询
    static crow::response handleSpatiotemporalQuery(const crow::request& req,
                                                   SpatiotemporalIndex& index,
                                                   HierarchicalMemoryStore& memory);
    
    // 图谱查询
    static crow::response handleGraphQuery(const crow::request& req,
                                          OntologyGraph& ontology);
    
    // 本体操作
    static crow::response handleOntologyCRUD(const crow::request& req,
                                            OntologyGraph& ontology);
    
    // 情境感知查询
    static crow::response handleContextualQuery(const crow::request& req,
                                               SpatiotemporalReasoner& reasoner,
                                               HierarchicalMemoryStore& memory);
    
    // 健康检查
    static crow::response handleHealth();
    
    // 统计信息
    static crow::response handleStats(OntologyGraph& ontology,
                                     HierarchicalMemoryStore& memory);
};

} // namespace pos
