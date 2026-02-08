/**
 * @file types.h
 * @brief 个人本体系统核心类型定义
 * 
 * 本文件定义了系统使用的所有核心数据类型、枚举和结构体
 * 采用现代C++20特性实现
 */

#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <array>
#include <chrono>
#include <optional>
#include <unordered_map>
#include <functional>
#include <variant>
#include <memory>
#include <format>
#include <compare>

namespace personal_ontology {

// =============================================================================
// 基础类型定义
// =============================================================================

/** 实体唯一标识符 */
using EntityId = std::string;

/** 记忆唯一标识符 */
using MemoryId = std::string;

/** 概念唯一标识符 */
using ConceptId = std::string;

/** 时间戳类型 (Unix时间戳, 毫秒精度) */
using Timestamp = int64_t;

/** 置信度分数 (0.0 - 1.0) */
using Confidence = double;

/** 向量嵌入类型 - 支持动态维度 */
using Embedding = std::vector<float>;

/** 属性值类型 - 支持多种类型的变体 */
using PropertyValue = std::variant<
    std::monostate,    // 空值
    bool,              // 布尔值
    int64_t,           // 整数
    double,            // 浮点数
    std::string,       // 字符串
    std::vector<std::string>  // 字符串列表
>;

/** 属性映射表 */
using Properties = std::unordered_map<std::string, PropertyValue>;

// =============================================================================
// 地理空间类型
// =============================================================================

/**
 * @brief 地理坐标点
 * 
 * 使用WGS84坐标系 (EPSG:4326)
 * 经度范围: -180 ~ 180
 * 纬度范围: -90 ~ 90
 */
struct GeoPoint {
    double longitude;  // 经度
    double latitude;   // 纬度
    std::optional<double> altitude;  // 可选的海拔高度(米)
    
    // 默认构造函数
    GeoPoint() : longitude(0.0), latitude(0.0), altitude(std::nullopt) {}
    
    // 完整构造函数
    GeoPoint(double lon, double lat, std::optional<double> alt = std::nullopt)
        : longitude(lon), latitude(lat), altitude(alt) {
        validate();
    }
    
    // 验证坐标有效性
    void validate() const {
        if (longitude < -180.0 || longitude > 180.0) {
            throw std::invalid_argument(std::format(
                "经度超出范围: {}, 应在 [-180, 180]", longitude));
        }
        if (latitude < -90.0 || latitude > 90.0) {
            throw std::invalid_argument(std::format(
                "纬度超出范围: {}, 应在 [-90, 90]", latitude));
        }
    }
    
    // C++20 三路比较运算符
    auto operator<=>(const GeoPoint& other) const = default;
    bool operator==(const GeoPoint& other) const = default;
    
    // 格式化为字符串
    [[nodiscard]] std::string toString() const {
        if (altitude.has_value()) {
            return std::format("({}, {}, {}m)", longitude, latitude, altitude.value());
        }
        return std::format("({}, {})", longitude, latitude);
    }
};

/**
 * @brief 地理边界框
 * 
 * 用于空间索引和查询
 */
struct GeoBoundingBox {
    double min_lon;
    double min_lat;
    double max_lon;
    double max_lat;
    
    [[nodiscard]] bool contains(const GeoPoint& point) const noexcept {
        return point.longitude >= min_lon && point.longitude <= max_lon &&
               point.latitude >= min_lat && point.latitude <= max_lat;
    }
    
    [[nodiscard]] bool intersects(const GeoBoundingBox& other) const noexcept {
        return !(other.min_lon > max_lon || other.max_lon < min_lon ||
                 other.min_lat > max_lat || other.max_lat < min_lat);
    }
};

// =============================================================================
// 枚举类型定义
// =============================================================================

/**
 * @brief 记忆类型枚举
 */
enum class MemoryType : uint8_t {
    EPISODIC,      // 情景记忆 - 具体事件
    SEMANTIC,      // 语义记忆 - 一般知识
    PROCEDURAL,    // 程序性记忆 - 技能和过程
    EMOTIONAL,     // 情感记忆 - 情感体验
    SPATIAL,       // 空间记忆 - 位置信息
    TEMPORAL,      // 时间记忆 - 时间相关信息
    CONVERSATION,  // 对话记忆 - 聊天记录
    DOCUMENT,      // 文档记忆 - 文件内容
    OBSERVATION,   // 观察记忆 - 环境感知
    REFLECTION     // 反思记忆 - 自我反思
};

/**
 * @brief 将MemoryType转换为字符串
 */
[[nodiscard]] inline std::string memoryTypeToString(MemoryType type) {
    switch (type) {
        case MemoryType::EPISODIC:     return "episodic";
        case MemoryType::SEMANTIC:     return "semantic";
        case MemoryType::PROCEDURAL:   return "procedural";
        case MemoryType::EMOTIONAL:    return "emotional";
        case MemoryType::SPATIAL:      return "spatial";
        case MemoryType::TEMPORAL:     return "temporal";
        case MemoryType::CONVERSATION: return "conversation";
        case MemoryType::DOCUMENT:     return "document";
        case MemoryType::OBSERVATION:  return "observation";
        case MemoryType::REFLECTION:   return "reflection";
        default: return "unknown";
    }
}

/**
 * @brief 从字符串解析MemoryType
 */
[[nodiscard]] inline MemoryType stringToMemoryType(const std::string& str) {
    static const std::unordered_map<std::string, MemoryType> map = {
        {"episodic", MemoryType::EPISODIC},
        {"semantic", MemoryType::SEMANTIC},
        {"procedural", MemoryType::PROCEDURAL},
        {"emotional", MemoryType::EMOTIONAL},
        {"spatial", MemoryType::SPATIAL},
        {"temporal", MemoryType::TEMPORAL},
        {"conversation", MemoryType::CONVERSATION},
        {"document", MemoryType::DOCUMENT},
        {"observation", MemoryType::OBSERVATION},
        {"reflection", MemoryType::REFLECTION}
    };
    auto it = map.find(str);
    if (it == map.end()) {
        throw std::invalid_argument(std::format("未知的记忆类型: {}", str));
    }
    return it->second;
}

/**
 * @brief 概念类型枚举
 */
enum class ConceptType : uint8_t {
    ENTITY,        // 实体 - 人、地点、物品
    EVENT,         // 事件 - 发生的活动
    ACTION,        // 动作 - 可执行的操作
    ATTRIBUTE,     // 属性 - 特征和性质
    RELATION,      // 关系 - 概念间联系
    CATEGORY,      // 类别 - 分类概念
    ABSTRACT,      // 抽象 - 抽象概念
    TEMPORAL,      // 时间 - 时间相关概念
    SPATIAL        // 空间 - 空间相关概念
};

[[nodiscard]] inline std::string conceptTypeToString(ConceptType type) {
    switch (type) {
        case ConceptType::ENTITY:    return "entity";
        case ConceptType::EVENT:     return "event";
        case ConceptType::ACTION:    return "action";
        case ConceptType::ATTRIBUTE: return "attribute";
        case ConceptType::RELATION:  return "relation";
        case ConceptType::CATEGORY:  return "category";
        case ConceptType::ABSTRACT:  return "abstract";
        case ConceptType::TEMPORAL:  return "temporal";
        case ConceptType::SPATIAL:   return "spatial";
        default: return "unknown";
    }
}

/**
 * @brief 关系类型枚举
 */
enum class RelationType : uint8_t {
    // 层次关系
    IS_A,          // 是一个 (猫 IS_A 动物)
    PART_OF,       // 部分 (轮子 PART_OF 汽车)
    INSTANCE_OF,   // 实例 (我的猫 INSTANCE_OF 猫)
    
    // 关联关系
    RELATED_TO,    // 相关 (医生 RELATED_TO 医院)
    SIMILAR_TO,    // 相似 (猫 SIMILAR_TO 老虎)
    OPPOSITE_OF,   // 相反 (热 OPPOSITE_OF 冷)
    
    // 因果关系
    CAUSES,        // 导致 (吸烟 CAUSES 癌症)
    PREVENTS,      // 防止 (疫苗 PREVENTS 疾病)
    ENABLES,       // 使能 (钥匙 ENABLES 开门)
    
    // 时空关系
    LOCATED_AT,    // 位于 (商店 LOCATED_AT 街道)
    OCCURS_AT,     // 发生于 (会议 OCCURS_AT 时间)
    BEFORE,        // 之前 (早餐 BEFORE 午餐)
    AFTER,         // 之后 (午餐 AFTER 早餐)
    DURING,        // 期间 (演讲 DURING 会议)
    
    // 社交关系
    KNOWS,         // 认识 (我 KNOWS 朋友)
    WORKS_WITH,    // 共事 (同事 WORKS_WITH 同事)
    FAMILY_OF,     // 家人 (父母 FAMILY_OF 孩子)
    
    // 属性关系
    HAS_PROPERTY,  // 具有 (苹果 HAS_PROPERTY 红色)
    HAS_PART,      // 包含 (汽车 HAS_PART 引擎)
    HAS_FUNCTION,  // 功能 (手机 HAS_FUNCTION 通讯)
    
    // 自定义关系
    CUSTOM         // 自定义 (通过属性定义)
};

[[nodiscard]] inline std::string relationTypeToString(RelationType type) {
    switch (type) {
        case RelationType::IS_A:          return "is_a";
        case RelationType::PART_OF:       return "part_of";
        case RelationType::INSTANCE_OF:   return "instance_of";
        case RelationType::RELATED_TO:    return "related_to";
        case RelationType::SIMILAR_TO:    return "similar_to";
        case RelationType::OPPOSITE_OF:   return "opposite_of";
        case RelationType::CAUSES:        return "causes";
        case RelationType::PREVENTS:      return "prevents";
        case RelationType::ENABLES:       return "enables";
        case RelationType::LOCATED_AT:    return "located_at";
        case RelationType::OCCURS_AT:     return "occurs_at";
        case RelationType::BEFORE:        return "before";
        case RelationType::AFTER:         return "after";
        case RelationType::DURING:        return "during";
        case RelationType::KNOWS:         return "knows";
        case RelationType::WORKS_WITH:    return "works_with";
        case RelationType::FAMILY_OF:     return "family_of";
        case RelationType::HAS_PROPERTY:  return "has_property";
        case RelationType::HAS_PART:      return "has_part";
        case RelationType::HAS_FUNCTION:  return "has_function";
        case RelationType::CUSTOM:        return "custom";
        default: return "unknown";
    }
}

// =============================================================================
// 核心数据结构
// =============================================================================

/**
 * @brief 源信息结构
 * 
 * 记录记忆的来源和原始数据
 */
struct SourceInfo {
    std::string source_type;      // 来源类型 (file, api, sensor, user_input等)
    std::string source_id;        // 来源标识
    std::string original_text;    // 原始文本内容
    std::string source_url;       // 来源URL (可选)
    Properties metadata;          // 额外元数据
    
    [[nodiscard]] bool isValid() const noexcept {
        return !source_type.empty() && !source_id.empty();
    }
};

/**
 * @brief 情感标签结构
 * 
 * 记录与记忆相关的情感信息
 */
struct EmotionalTag {
    std::string emotion;          // 情感类型 (happy, sad, angry等)
    double intensity;             // 强度 0.0-1.0
    Timestamp timestamp;          // 发生时间
    std::optional<std::string> context;  // 情感上下文
};

/**
 * @brief 记忆痕迹结构体
 * 
 * 这是系统的核心数据结构, 代表一个单独的记忆单元
 * 包含内容、元数据、时空信息和向量嵌入
 */
struct MemoryTrace {
    // 基础标识
    MemoryId id;                  // 唯一标识符 (UUID)
    EntityId entity_id;           // 所属实体ID
    MemoryType memory_type;       // 记忆类型
    
    // 时间信息
    Timestamp created_at;         // 创建时间
    Timestamp updated_at;         // 最后更新时间
    std::optional<Timestamp> event_time;    // 事件实际发生时间
    std::optional<Timestamp> expiration;    // 过期时间 (可选)
    
    // 空间信息
    std::optional<GeoPoint> location;       // 地理位置
    std::optional<std::string> location_name; // 地点名称
    
    // 内容信息
    std::string content;          // 记忆内容 (文本)
    std::vector<std::string> tags; // 标签列表
    std::vector<EmotionalTag> emotions; // 情感标签
    
    // 向量嵌入
    std::optional<Embedding> embedding;     // 语义向量
    std::optional<Embedding> multimodal_embedding; // 多模态向量 (可选)
    
    // 源信息
    SourceInfo source;            // 来源信息
    
    // 关联信息
    std::vector<ConceptId> related_concepts; // 相关概念ID
    std::vector<MemoryId> related_memories;  // 相关记忆ID
    
    // 元数据和属性
    Properties properties;        // 自定义属性
    Confidence confidence;        // 置信度 0.0-1.0
    uint32_t access_count;        // 访问次数 (用于记忆强化)
    Timestamp last_accessed;      // 最后访问时间
    
    // 分层存储信息
    uint8_t memory_layer;         // 存储层级 (0=工作记忆, 1=短期, 2=长期)
    float importance_score;       // 重要性评分 (用于记忆巩固)
    
    // 默认构造函数
    MemoryTrace() 
        : id("")
        , entity_id("")
        , memory_type(MemoryType::EPISODIC)
        , created_at(0)
        , updated_at(0)
        , confidence(1.0)
        , access_count(0)
        , last_accessed(0)
        , memory_layer(0)
        , importance_score(0.5f) {}
    
    // 创建当前时间戳
    [[nodiscard]] static Timestamp now() {
        return std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
    }
    
    // 更新访问统计
    void recordAccess() noexcept {
        access_count++;
        last_accessed = now();
    }
    
    // 检查是否过期
    [[nodiscard]] bool isExpired() const noexcept {
        if (!expiration.has_value()) return false;
        return now() > expiration.value();
    }
};

/**
 * @brief 概念关系结构
 */
struct ConceptRelation {
    RelationType type;            // 关系类型
    ConceptId target_concept_id;  // 目标概念ID
    Confidence confidence;        // 关系置信度
    Properties attributes;        // 关系属性
    Timestamp created_at;         // 创建时间
    
    // 自定义关系名称 (当type为CUSTOM时使用)
    std::optional<std::string> custom_name;
    
    [[nodiscard]] std::string getName() const {
        if (type == RelationType::CUSTOM && custom_name.has_value()) {
            return custom_name.value();
        }
        return relationTypeToString(type);
    }
};

/**
 * @brief 本体概念结构体
 * 
 * 代表本体图谱中的一个概念节点
 */
struct OntologyConcept {
    ConceptId id;                 // 唯一标识符
    std::string name;             // 概念名称
    std::vector<std::string> aliases; // 别名列表
    ConceptType concept_type;     // 概念类型
    
    // 描述信息
    std::string description;      // 概念描述
    std::vector<std::string> definitions; // 定义列表
    
    // 向量表示
    std::optional<Embedding> embedding; // 语义向量
    
    // 关系
    std::vector<ConceptRelation> outgoing_relations; // 出边关系
    std::vector<ConceptRelation> incoming_relations; // 入边关系 (索引用)
    
    // 属性
    Properties properties;        // 概念属性
    
    // 元数据
    Timestamp created_at;         // 创建时间
    Timestamp updated_at;         // 更新时间
    Confidence confidence;        // 置信度
    std::vector<std::string> sources; // 知识来源
    
    // 统计信息
    uint32_t reference_count;     // 引用计数
    
    OntologyConcept() 
        : id("")
        , name("")
        , concept_type(ConceptType::ENTITY)
        , created_at(0)
        , updated_at(0)
        , confidence(1.0)
        , reference_count(0) {}
    
    // 添加出边关系
    void addOutgoingRelation(const ConceptRelation& rel) {
        outgoing_relations.push_back(rel);
    }
    
    // 添加入边关系
    void addIncomingRelation(const ConceptRelation& rel) {
        incoming_relations.push_back(rel);
    }
    
    // 查找特定类型的关系
    [[nodiscard]] std::vector<ConceptRelation> findRelations(RelationType type) const {
        std::vector<ConceptRelation> result;
        for (const auto& rel : outgoing_relations) {
            if (rel.type == type) {
                result.push_back(rel);
            }
        }
        return result;
    }
};

// =============================================================================
// 查询和结果类型
// =============================================================================

/**
 * @brief 记忆查询条件
 */
struct MemoryQuery {
    std::optional<MemoryType> type_filter;           // 类型过滤
    std::optional<std::string> content_keyword;      // 内容关键词
    std::optional<Timestamp> time_start;             // 时间范围开始
    std::optional<Timestamp> time_end;               // 时间范围结束
    std::optional<GeoBoundingBox> spatial_bounds;    // 空间范围
    std::optional<Embedding> semantic_query;         // 语义查询向量
    std::vector<std::string> tags_filter;            // 标签过滤
    std::vector<ConceptId> concept_filter;           // 概念过滤
    std::optional<double> min_confidence;            // 最小置信度
    
    // 分页
    uint32_t limit = 100;
    uint32_t offset = 0;
    
    // 排序选项
    enum class SortBy {
        TIME_DESC,    // 时间降序
        TIME_ASC,     // 时间升序
        RELEVANCE,    // 相关性
        IMPORTANCE,   // 重要性
        ACCESS_COUNT  // 访问次数
    };
    SortBy sort_by = SortBy::TIME_DESC;
};

/**
 * @brief 查询结果
 */
template<typename T>
struct QueryResult {
    std::vector<T> items;         // 结果项
    uint32_t total_count;         // 总数量
    bool has_more;                // 是否有更多结果
    
    QueryResult() : total_count(0), has_more(false) {}
};

using MemoryQueryResult = QueryResult<MemoryTrace>;
using ConceptQueryResult = QueryResult<OntologyConcept>;

/**
 * @brief 搜索结果项 (带分数)
 */
template<typename T>
struct ScoredItem {
    T item;
    double score;  // 相似度/相关性分数
    
    bool operator>(const ScoredItem& other) const {
        return score > other.score;
    }
};

// =============================================================================
// 错误处理类型
// =============================================================================

/**
 * @brief 系统错误码
 */
enum class ErrorCode : uint16_t {
    SUCCESS = 0,
    
    // 存储错误
    STORAGE_NOT_FOUND = 1001,
    STORAGE_ALREADY_EXISTS = 1002,
    STORAGE_WRITE_ERROR = 1003,
    STORAGE_READ_ERROR = 1004,
    STORAGE_SERIALIZATION_ERROR = 1005,
    
    // 查询错误
    QUERY_INVALID_PARAMS = 2001,
    QUERY_EXECUTION_ERROR = 2002,
    QUERY_TIMEOUT = 2003,
    
    // ML服务错误
    ML_SERVICE_UNAVAILABLE = 3001,
    ML_SERVICE_TIMEOUT = 3002,
    ML_SERVICE_ERROR = 3003,
    EMBEDDING_GENERATION_FAILED = 3004,
    
    // 配置错误
    CONFIG_NOT_FOUND = 4001,
    CONFIG_PARSE_ERROR = 4002,
    CONFIG_INVALID_VALUE = 4003,
    
    // 运行时错误
    INVALID_ARGUMENT = 5001,
    NOT_IMPLEMENTED = 5002,
    INTERNAL_ERROR = 5003
};

/**
 * @brief 结果类型 (类似Rust的Result)
 */
template<typename T>
class Result {
public:
    Result(T value) : value_(std::move(value)), is_ok_(true) {}
    Result(ErrorCode code, std::string message) 
        : error_code_(code), error_message_(std::move(message)), is_ok_(false) {}
    
    [[nodiscard]] bool isOk() const noexcept { return is_ok_; }
    [[nodiscard]] bool isError() const noexcept { return !is_ok_; }
    
    [[nodiscard]] T& value() & { 
        if (!is_ok_) throw std::runtime_error("Result contains error");
        return value_; 
    }
    [[nodiscard]] const T& value() const& { 
        if (!is_ok_) throw std::runtime_error("Result contains error");
        return value_; 
    }
    [[nodiscard]] T&& value() && { 
        if (!is_ok_) throw std::runtime_error("Result contains error");
        return std::move(value_); 
    }
    
    [[nodiscard]] ErrorCode errorCode() const { return error_code_; }
    [[nodiscard]] const std::string& errorMessage() const { return error_message_; }
    
private:
    T value_;
    ErrorCode error_code_;
    std::string error_message_;
    bool is_ok_;
};

// =============================================================================
// 工具函数
// =============================================================================

/**
 * @brief 生成UUID v4
 */
[[nodiscard]] inline std::string generateUUID() {
    static std::random_device rd;
    static std::mt19937_64 gen(rd());
    static std::uniform_int_distribution<> dis(0, 15);
    static std::uniform_int_distribution<> dis2(8, 11);
    
    std::stringstream ss;
    ss << std::hex;
    
    for (int i = 0; i < 8; i++) {
        ss << dis(gen);
    }
    ss << "-";
    for (int i = 0; i < 4; i++) {
        ss << dis(gen);
    }
    ss << "-4";  // UUID版本4
    for (int i = 0; i < 3; i++) {
        ss << dis(gen);
    }
    ss << "-";
    ss << dis2(gen);  // variant
    for (int i = 0; i < 3; i++) {
        ss << dis(gen);
    }
    ss << "-";
    for (int i = 0; i < 12; i++) {
        ss << dis(gen);
    }
    
    return ss.str();
}

} // namespace personal_ontology