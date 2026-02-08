#pragma once

#include "core/common/types.h"
#include <memory>
#include <mutex>

namespace pos {

// 前向声明
class RocksDBStorage;

// 本体概念结构
struct OntologyConcept {
    ConceptId id;
    ConceptType type;
    std::string label;
    std::string description;
    
    // 属性存储
    using PropertyValue = std::variant<std::string, int, double, bool>;
    std::unordered_map<std::string, PropertyValue> properties;
    
    // 关系
    struct Relation {
        RelationType type;
        ConceptId target;
        float weight{1.0};
        std::optional<std::string> temporal_constraint;
        std::optional<Timestamp> valid_from;
        std::optional<Timestamp> valid_until;
    };
    std::vector<Relation> relations;
    
    // 记忆绑定（双向绑定）
    std::unordered_set<MemoryId> bound_memories;
    
    // 元数据
    Timestamp created_at;
    Timestamp updated_at;
    float confidence{1.0};
    std::string source{"inferred"};  // manual, inferred, imported, learned
    int access_count{0};
    
    // 方法
    void bindMemory(const MemoryId& memory_id);
    void unbindMemory(const MemoryId& memory_id);
    void addRelation(const Relation& rel);
    bool hasRelationTo(const ConceptId& target, RelationType type) const;
    
    // 序列化
    std::string toJson() const;
    static OntologyConcept fromJson(const std::string& json_str);
};

// 本体图谱管理器
class OntologyGraph {
public:
    explicit OntologyGraph(const std::string& storage_path);
    ~OntologyGraph();
    
    // 禁止拷贝
    OntologyGraph(const OntologyGraph&) = delete;
    OntologyGraph& operator=(const OntologyGraph&) = delete;
    
    // CRUD操作
    ConceptId createConcept(const std::string& label, ConceptType type, 
                           const std::string& source = "manual");
    std::optional<OntologyConcept> getConcept(const ConceptId& id) const;
    std::vector<OntologyConcept> findConceptsByLabel(const std::string& label, 
                                                       int limit = 10) const;
    std::vector<OntologyConcept> findConceptsByType(ConceptType type, 
                                                       int limit = 100) const;
    bool updateConcept(const OntologyConcept& concept);
    bool deleteConcept(const ConceptId& id);
    
    // 关系操作
    bool addRelation(const ConceptId& from, const ConceptId& to, 
                     RelationType type, float weight = 1.0,
                     const std::optional<std::string>& temporal = std::nullopt);
    bool removeRelation(const ConceptId& from, const ConceptId& to,
                        RelationType type);
    std::vector<OntologyConcept> getRelatedConcepts(
        const ConceptId& concept, 
        RelationType type,
        int depth = 1
    ) const;
    std::vector<OntologyConcept> getAllRelated(const ConceptId& concept, 
                                                 int depth = 1) const;
    
    // 语义扩展
    std::vector<OntologyConcept> expandSemantically(
        const ConceptId& concept,
        int depth = 2,
        float min_relation_weight = 0.5
    ) const;
    
    // 记忆增强本体
    void augmentFromPattern(const std::vector<MemoryId>& memories,
                           const std::string& pattern_type);
    void bindMemoryToConcept(const ConceptId& concept, const MemoryId& memory);
    void unbindMemoryFromConcept(const ConceptId& concept, const MemoryId& memory);
    
    // 查询
    std::vector<OntologyConcept> query(const std::string& query_text,
                                       int limit = 10) const;
    std::vector<OntologyConcept> findPath(const ConceptId& from,
                                           const ConceptId& to,
                                           int max_depth = 5) const;
    
    // 统计
    size_t getConceptCount() const;
    size_t getRelationCount() const;
    
    // 批量操作
    bool importFromJson(const std::string& json_file);
    std::string exportToJson() const;

private:
    class Impl;
    std::unique_ptr<Impl> pimpl_;
    mutable std::mutex mutex_;
};

} // namespace pos
