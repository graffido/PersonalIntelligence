#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <chrono>
#include <memory>
#include <optional>
#include <variant>
#include <functional>

namespace pos {

// 唯一标识符类型
using EntityId = std::string;
using MemoryId = std::string;
using ConceptId = std::string;

// 时间类型
using Timestamp = std::chrono::system_clock::time_point;
using Duration = std::chrono::seconds;

// 空间类型
struct GeoPoint {
    double latitude{0.0};
    double longitude{0.0};
    double altitude{0.0};
    
    bool operator==(const GeoPoint& other) const {
        return std::abs(latitude - other.latitude) < 1e-9 
            && std::abs(longitude - other.longitude) < 1e-9;
    }
    
    double distanceTo(const GeoPoint& other) const;
};

struct GeoBoundingBox {
    GeoPoint min;
    GeoPoint max;
    
    bool contains(const GeoPoint& p) const;
};

// 向量类型
using Embedding = std::vector<float>;

// 情感标签
struct EmotionalTag {
    std::string emotion;      // happy/sad/anxious/excited等
    float valence{0.0};       // -1 (negative) to +1 (positive)
    float arousal{0.0};       // 0 (calm) to 1 (excited)
    float dominance{0.5};     // 0 (controlled) to 1 (in-control)
};

// 记忆类型枚举
enum class MemoryType {
    EPISODIC,      // 情景记忆 - 具体事件
    SEMANTIC,      // 语义记忆 - 事实知识
    PROCEDURAL,    // 程序记忆 - 技能和习惯
    SENSORY,       // 感官记忆 - 原始感知
    WORKING        // 工作记忆 - 当前焦点
};

// 本体概念类型
enum class ConceptType {
    PERSON,
    PLACE,
    EVENT,
    ARTIFACT,
    CONCEPT,
    ORGANIZATION,
    TIME,
    RELATION,
    UNKNOWN
};

// 关系类型
enum class RelationType {
    KNOWS,         // 认识
    LOCATED_AT,    // 位于
    PARTICIPATES_IN, // 参与
    CREATED,       // 创建
    OWNS,          // 拥有
    RELATED_TO,    // 相关
    PRECEDES,      // 先于
    CAUSES,        // 导致
    PART_OF,       // 部分
    SIMILAR_TO,    // 相似
    CUSTOM         // 自定义
};

// 时间关系
enum class TemporalRelation {
    BEFORE,
    AFTER,
    DURING,
    CONTAINS,
    OVERLAPS,
    MEETS,
    EQUALS
};

// 概念类别到字符串
std::string conceptTypeToString(ConceptType type);
ConceptType stringToConceptType(const std::string& str);

// 关系类型到字符串
std::string relationTypeToString(RelationType type);
RelationType stringToRelationType(const std::string& str);

// 生成UUID
std::string generateUUID();

// 时间工具函数
Timestamp parseTimestamp(const std::string& iso_string);
std::string timestampToString(const Timestamp& ts);
std::string getTimeOfDay(const Timestamp& ts);      // morning/afternoon/evening/night
std::string getDayType(const Timestamp& ts);        // weekday/weekend

} // namespace pos
