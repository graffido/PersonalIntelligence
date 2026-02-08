#include "api/http_server.h"
#include "core/pos_core_engine.h"
#include <nlohmann/json.hpp>
#include <iostream>

namespace pos {

using json = nlohmann::json;

// 全局核心引擎实例
std::unique_ptr<POSCoreEngine> g_engine;

// 工具函数
json memoryToJson(const MemoryTrace& m) {
    json j;
    j["id"] = m.id;
    j["content"] = m.content.raw_text;
    j["timestamp"] = timestampToString(m.timestamp);
    if (m.location) {
        j["location"] = {
            {"lat", m.location->latitude},
            {"lng", m.location->longitude}
        };
    }
    return j;
}

json entityToJson(const ExtractedEntity& e) {
    json j;
    j["text"] = e.text;
    j["label"] = e.label;
    j["normalized"] = e.normalized_form;
    j["confidence"] = e.confidence;
    return j;
}

json recommendationToJson(const UnifiedProcessResult::Recommendation& r) {
    json j;
    j["type"] = r.type;
    j["title"] = r.title;
    j["description"] = r.description;
    j["confidence"] = r.confidence;
    j["priority"] = r.priority;
    j["reason"] = r.reason;
    return j;
}

// 处理器实现
crow::response handleUnifiedInput(const crow::request& req) {
    try {
        auto body = json::parse(req.body);
        
        UnifiedInputRequest request;
        request.raw_text = body["text"];
        
        if (body.contains("time")) {
            request.explicit_time = parseTimestamp(body["time"]);
        }
        if (body.contains("location")) {
            request.explicit_location = GeoPoint{
                body["location"]["lat"],
                body["location"]["lng"]
            };
        }
        
        // 调用核心引擎
        auto result = g_engine->processInput(request);
        
        // 构建响应
        json response;
        response["success"] = true;
        response["memory_id"] = result.memory_id;
        
        response["entities"] = json::array();
        for (const auto& e : result.entities) {
            response["entities"].push_back(entityToJson(e));
        }
        
        response["reasoning_results"] = json::array();
        for (const auto& r : result.reasoning_results) {
            json rj;
            rj["rule"] = r.rule_name;
            rj["type"] = r.type;
            rj["description"] = r.description;
            rj["confidence"] = r.confidence;
            response["reasoning_results"].push_back(rj);
        }
        
        response["recommendations"] = json::array();
        for (const auto& r : result.recommendations) {
            response["recommendations"].push_back(recommendationToJson(r));
        }
        
        response["stats"] = {
            {"new_concepts", result.new_concepts_created},
            {"linked_concepts", result.existing_concepts_linked}
        };
        
        return crow::response(200, response.dump(2));
        
    } catch (const std::exception& e) {
        return crow::response(500, json{{"error", e.what()}}.dump());
    }
}

crow::response handleQuery(const crow::request& req) {
    try {
        auto body = json::parse(req.body);
        
        QueryRequest request;
        request.query_text = body["text"];
        request.query_type = body.value("type", "auto");
        
        if (body.contains("time_hint")) {
            request.time_hint = parseTimestamp(body["time_hint"]);
        }
        
        auto result = g_engine->query(request);
        
        json response;
        response["strategy"] = result.strategy_used;
        response["results"] = json::array();
        
        for (const auto& m : result.direct_matches) {
            response["results"].push_back(memoryToJson(m));
        }
        
        response["inferred_count"] = result.inferred_matches.size();
        response["reasoning_applied"] = !result.reasoning_path.empty();
        
        return crow::response(200, response.dump(2));
        
    } catch (const std::exception& e) {
        return crow::response(500, json{{"error", e.what()}}.dump());
    }
}

crow::response handleRecommendations(const crow::request& req) {
    try {
        auto body = json::parse(req.body);
        
        SituationContext ctx;
        ctx.current_time = std::chrono::system_clock::now();
        
        if (body.contains("location")) {
            ctx.current_location = GeoPoint{
                body["location"]["lat"],
                body["location"]["lng"]
            };
        }
        
        int limit = body.value("limit", 5);
        auto recommendations = g_engine->generateRecommendations(ctx, limit);
        
        json response;
        response["recommendations"] = json::array();
        for (const auto& r : recommendations) {
            response["recommendations"].push_back(recommendationToJson(r));
        }
        response["count"] = recommendations.size();
        
        return crow::response(200, response.dump(2));
        
    } catch (const std::exception& e) {
        return crow::response(500, json{{"error", e.what()}}.dump());
    }
}

crow::response handlePredictions(const crow::request& req) {
    try {
        auto query = req.url_params.get("hours");
        int hours = query ? std::stoi(query) : 24;
        
        auto predictions = g_engine->predictNextEvents(
            std::chrono::system_clock::now(),
            hours
        );
        
        json response;
        response["predictions"] = json::array();
        for (const auto& p : predictions) {
            json pj;
            pj["event_type"] = p.event_type;
            pj["confidence"] = p.confidence;
            pj["description"] = p.description;
            response["predictions"].push_back(pj);
        }
        
        return crow::response(200, response.dump(2));
        
    } catch (const std::exception& e) {
        return crow::response(500, json{{"error", e.what()}}.dump());
    }
}

crow::response handleStats() {
    auto stats = g_engine->getStats();
    
    json j;
    j["memories"] = stats.memory_count;
    j["concepts"] = stats.concept_count;
    j["relations"] = stats.relation_count;
    j["status"] = "ok";
    
    return crow::response(200, j.dump());
}

crow::response handleHealth() {
    return crow::response(200, json{
        {"status", "ok"},
        {"version", "2.0.0"},
        {"core", "pos_core"}
    }.dump());
}

// 启动服务器
void runServer(const std::string& data_dir,
               const std::string& ml_endpoint,
               int port) {
    
    // 初始化核心引擎
    MLServiceConfig ml_config;
    ml_config.endpoint = ml_endpoint;
    ml_config.embedding.dimension = 384;
    
    g_engine = std::make_unique<POSCoreEngine>(data_dir, ml_config);
    
    // 创建Crow应用
    crow::SimpleApp app;
    
    // CORS
    app.use([](crow::request& req, crow::response& res, crow::context& ctx) {
        res.add_header("Access-Control-Allow-Origin", "*");
        res.add_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
        res.add_header("Access-Control-Allow-Headers", "Content-Type");
        if (req.method == crow::HTTPMethod::OPTIONS) {
            res.code = 200;
            res.end();
            return;
        }
        ctx.next();
    });
    
    // 路由
    CROW_ROUTE(app, "/health").methods(crow::HTTPMethod::GET)(handleHealth);
    CROW_ROUTE(app, "/api/v1/stats").methods(crow::HTTPMethod::GET)(handleStats);
    CROW_ROUTE(app, "/api/v1/input").methods(crow::HTTPMethod::POST)(handleUnifiedInput);
    CROW_ROUTE(app, "/api/v1/query").methods(crow::HTTPMethod::POST)(handleQuery);
    CROW_ROUTE(app, "/api/v1/recommendations").methods(crow::HTTPMethod::POST)(handleRecommendations);
    CROW_ROUTE(app, "/api/v1/predictions").methods(crow::HTTPMethod::GET)(handlePredictions);
    
    std::cout << "[POS Server] Starting on port " << port << std::endl;
    std::cout << "[POS Server] Data directory: " << data_dir << std::endl;
    std::cout << "[POS Server] ML endpoint: " << ml_endpoint << std::endl;
    
    app.port(port).multithreaded().run();
}

} // namespace pos

// 主函数
int main(int argc, char* argv[]) {
    std::string data_dir = "./data";
    std::string ml_endpoint = "http://localhost:8000";
    int port = 8080;
    
    if (argc > 1) data_dir = argv[1];
    if (argc > 2) ml_endpoint = argv[2];
    if (argc > 3) port = std::stoi(argv[3]);
    
    pos::runServer(data_dir, ml_endpoint, port);
    
    return 0;
}
