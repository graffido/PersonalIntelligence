#pragma once

#include "core/common/types.h"
#include <string>
#include <vector>

namespace pos {

// NER实体
struct Entity {
    std::string text;
    std::string label;        // PERSON, ORG, GPE, EVENT, DATE, TIME, etc.
    size_t start_pos{0};
    size_t end_pos{0};
    float confidence{1.0};
};

// 关系
struct Relation {
    std::string subject;
    std::string predicate;
    std::string object;
    float confidence{0.0};
};

// ML服务客户端配置
struct MLServiceConfig {
    std::string endpoint{"http://localhost:8000"};
    int timeout_ms{5000};
    
    struct {
        std::string name{"text-embedding-3-small"};
        int dimension{1536};
    } embedding;
    
    struct {
        std::string name{"spacy-en-core-web-lg"};
        std::vector<std::string> entity_types;
    } ner;
};

// ML服务客户端
class MLServiceClient {
public:
    explicit MLServiceClient(const MLServiceConfig& config);
    ~MLServiceClient();
    
    // 健康检查
    bool healthCheck() const;
    
    // Embedding
    Embedding embed(const std::string& text) const;
    std::vector<Embedding> embedBatch(const std::vector<std::string>& texts) const;
    
    // NER实体抽取
    std::vector<Entity> extractEntities(const std::string& text) const;
    
    // 关系抽取
    std::vector<Relation> extractRelations(
        const std::string& text,
        const std::vector<Entity>& entities
    ) const;
    
    // 文本生成
    std::string generate(
        const std::string& prompt,
        float temperature = 0.7f,
        int max_tokens = 500
    ) const;
    
    // 批量处理
    struct BatchProcessResult {
        std::vector<Entity> entities;
        Embedding embedding;
        std::string summary;
    };
    
    BatchProcessResult processDocument(const std::string& text) const;

private:
    class Impl;
    std::unique_ptr<Impl> pimpl_;
};

} // namespace pos
