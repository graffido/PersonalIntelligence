#include "ml_service_client.h"
#include <curl/curl.h>
#include <nlohmann/json.hpp>
#include <sstream>
#include <iostream>

namespace pos {

using json = nlohmann::json;

// CURL写回调
static size_t WriteCallback(void* contents, size_t size, size_t nmemb, void* userp) {
    ((std::string*)userp)->append((char*)contents, size * nmemb);
    return size * nmemb;
}

class MLServiceClient::Impl {
public:
    MLServiceConfig config_;
    CURL* curl_;
    curl_slist* headers_;
    
    explicit Impl(const MLServiceConfig& config) : config_(config) {
        curl_global_init(CURL_GLOBAL_ALL);
        curl_ = curl_easy_init();
        
        headers_ = nullptr;
        headers_ = curl_slist_append(headers_, "Content-Type: application/json");
        
        curl_easy_setopt(curl_, CURLOPT_HTTPHEADER, headers_);
        curl_easy_setopt(curl_, CURLOPT_TIMEOUT_MS, config.timeout_ms);
    }
    
    ~Impl() {
        curl_slist_free_all(headers_);
        curl_easy_cleanup(curl_);
        curl_global_cleanup();
    }
    
    std::string post(const std::string& path, const std::string& body) const {
        std::string url = config_.endpoint + path;
        std::string response;
        
        curl_easy_setopt(curl_, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl_, CURLOPT_POSTFIELDS, body.c_str());
        curl_easy_setopt(curl_, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl_, CURLOPT_WRITEDATA, &response);
        
        CURLcode res = curl_easy_perform(curl_);
        
        if (res != CURLE_OK) {
            throw std::runtime_error(std::string("HTTP request failed: ") + 
                                   curl_easy_strerror(res));
        }
        
        long http_code;
        curl_easy_getinfo(curl_, CURLINFO_RESPONSE_CODE, &http_code);
        
        if (http_code != 200) {
            throw std::runtime_error("HTTP error: " + std::to_string(http_code) + 
                                   ", response: " + response);
        }
        
        return response;
    }
    
    std::string get(const std::string& path) const {
        std::string url = config_.endpoint + path;
        std::string response;
        
        curl_easy_setopt(curl_, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl_, CURLOPT_HTTPGET, 1L);
        curl_easy_setopt(curl_, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl_, CURLOPT_WRITEDATA, &response);
        
        CURLcode res = curl_easy_perform(curl_);
        
        if (res != CURLE_OK) {
            throw std::runtime_error(std::string("HTTP request failed: ") + 
                                   curl_easy_strerror(res));
        }
        
        return response;
    }
};

MLServiceClient::MLServiceClient(const MLServiceConfig& config)
    : pimpl_(std::make_unique<Impl>(config)) {}

MLServiceClient::~MLServiceClient() = default;

bool MLServiceClient::healthCheck() const {
    try {
        auto response = pimpl_->get("/health");
        json j = json::parse(response);
        return j.value("status", "error") == "ok";
    } catch (...) {
        return false;
    }
}

Embedding MLServiceClient::embed(const std::string& text) const {
    json request;
    request["texts"] = json::array({text});
    
    auto response = pimpl_->post("/embed", request.dump());
    
    json j = json::parse(response);
    Embedding result;
    
    for (const auto& val : j["embeddings"][0]) {
        result.push_back(val.get<float>());
    }
    
    return result;
}

std::vector<Embedding> MLServiceClient::embedBatch(
    const std::vector<std::string>& texts) const {
    
    json request;
    request["texts"] = texts;
    
    auto response = pimpl_->post("/embed", request.dump());
    
    json j = json::parse(response);
    std::vector<Embedding> results;
    
    for (const auto& emb : j["embeddings"]) {
        Embedding e;
        for (const auto& val : emb) {
            e.push_back(val.get<float>());
        }
        results.push_back(e);
    }
    
    return results;
}

std::vector<Entity> MLServiceClient::extractEntities(const std::string& text) const {
    json request;
    request["text"] = text;
    
    auto response = pimpl_->post("/ner", request.dump());
    
    json j = json::parse(response);
    std::vector<Entity> results;
    
    for (const auto& ej : j) {
        Entity e;
        e.text = ej["text"];
        e.label = ej["label"];
        e.start_pos = ej["start"];
        e.end_pos = ej["end"];
        e.confidence = ej.value("confidence", 1.0f);
        results.push_back(e);
    }
    
    return results;
}

std::string MLServiceClient::generate(const std::string& prompt,
                                     float temperature,
                                     int max_tokens) const {
    json request;
    request["prompt"] = prompt;
    request["temperature"] = temperature;
    request["max_tokens"] = max_tokens;
    
    auto response = pimpl_->post("/generate", request.dump());
    
    json j = json::parse(response);
    return j.value("text", "");
}

MLServiceClient::BatchProcessResult MLServiceClient::processDocument(
    const std::string& text) const {
    
    BatchProcessResult result;
    
    // 并行处理 (简化：串行)
    result.entities = extractEntities(text);
    result.embedding = embed(text);
    
    // 生成摘要
    std::string summary_prompt = "Summarize the following text in one sentence: \"" + 
                                 text.substr(0, 500) + "...\"";
    result.summary = generate(summary_prompt, 0.5, 100);
    
    return result;
}

} // namespace pos
