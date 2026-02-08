#include "types.h"
#include <cmath>
#include <sstream>
#include <iomanip>
#include <random>
#include <chrono>

namespace pos {

// GeoPoint实现
double GeoPoint::distanceTo(const GeoPoint& other) const {
    // Haversine formula
    constexpr double R = 6371000; // 地球半径(米)
    double lat1 = latitude * M_PI / 180.0;
    double lat2 = other.latitude * M_PI / 180.0;
    double deltaLat = (other.latitude - latitude) * M_PI / 180.0;
    double deltaLon = (other.longitude - longitude) * M_PI / 180.0;
    
    double a = std::sin(deltaLat/2) * std::sin(deltaLat/2) +
               std::cos(lat1) * std::cos(lat2) *
               std::sin(deltaLon/2) * std::sin(deltaLon/2);
    double c = 2 * std::atan2(std::sqrt(a), std::sqrt(1-a));
    
    return R * c;
}

bool GeoBoundingBox::contains(const GeoPoint& p) const {
    return p.latitude >= min.latitude && p.latitude <= max.latitude &&
           p.longitude >= min.longitude && p.longitude <= max.longitude;
}

// 概念类型转换
std::string conceptTypeToString(ConceptType type) {
    switch(type) {
        case ConceptType::PERSON: return "PERSON";
        case ConceptType::PLACE: return "PLACE";
        case ConceptType::EVENT: return "EVENT";
        case ConceptType::ARTIFACT: return "ARTIFACT";
        case ConceptType::CONCEPT: return "CONCEPT";
        case ConceptType::ORGANIZATION: return "ORGANIZATION";
        case ConceptType::TIME: return "TIME";
        case ConceptType::RELATION: return "RELATION";
        default: return "UNKNOWN";
    }
}

ConceptType stringToConceptType(const std::string& str) {
    if(str == "PERSON") return ConceptType::PERSON;
    if(str == "PLACE") return ConceptType::PLACE;
    if(str == "EVENT") return ConceptType::EVENT;
    if(str == "ARTIFACT") return ConceptType::ARTIFACT;
    if(str == "CONCEPT") return ConceptType::CONCEPT;
    if(str == "ORGANIZATION") return ConceptType::ORGANIZATION;
    if(str == "TIME") return ConceptType::TIME;
    if(str == "RELATION") return ConceptType::RELATION;
    return ConceptType::UNKNOWN;
}

// 关系类型转换
std::string relationTypeToString(RelationType type) {
    switch(type) {
        case RelationType::KNOWS: return "KNOWS";
        case RelationType::LOCATED_AT: return "LOCATED_AT";
        case RelationType::PARTICIPATES_IN: return "PARTICIPATES_IN";
        case RelationType::CREATED: return "CREATED";
        case RelationType::OWNS: return "OWNS";
        case RelationType::RELATED_TO: return "RELATED_TO";
        case RelationType::PRECEDES: return "PRECEDES";
        case RelationType::CAUSES: return "CAUSES";
        case RelationType::PART_OF: return "PART_OF";
        case RelationType::SIMILAR_TO: return "SIMILAR_TO";
        default: return "CUSTOM";
    }
}

RelationType stringToRelationType(const std::string& str) {
    if(str == "KNOWS") return RelationType::KNOWS;
    if(str == "LOCATED_AT") return RelationType::LOCATED_AT;
    if(str == "PARTICIPATES_IN") return RelationType::PARTICIPATES_IN;
    if(str == "CREATED") return RelationType::CREATED;
    if(str == "OWNS") return RelationType::OWNS;
    if(str == "RELATED_TO") return RelationType::RELATED_TO;
    if(str == "PRECEDES") return RelationType::PRECEDES;
    if(str == "CAUSES") return RelationType::CAUSES;
    if(str == "PART_OF") return RelationType::PART_OF;
    if(str == "SIMILAR_TO") return RelationType::SIMILAR_TO;
    return RelationType::CUSTOM;
}

// UUID生成
std::string generateUUID() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_int_distribution<> dis(0, 15);
    static std::uniform_int_distribution<> dis2(8, 11);
    
    std::stringstream ss;
    int i;
    ss << std::hex;
    for (i = 0; i < 8; i++) {
        ss << dis(gen);
    }
    ss << "-";
    for (i = 0; i < 4; i++) {
        ss << dis(gen);
    }
    ss << "-4";
    for (i = 0; i < 3; i++) {
        ss << dis(gen);
    }
    ss << "-";
    ss << dis2(gen);
    for (i = 0; i < 3; i++) {
        ss << dis(gen);
    }
    ss << "-";
    for (i = 0; i < 12; i++) {
        ss << dis(gen);
    }
    return ss.str();
}

// 时间解析
Timestamp parseTimestamp(const std::string& iso_string) {
    std::tm tm = {};
    std::stringstream ss(iso_string);
    ss >> std::get_time(&tm, "%Y-%m-%dT%H:%M:%S");
    if (ss.fail()) {
        // 尝试其他格式
        ss.clear();
        ss.str(iso_string);
        ss >> std::get_time(&tm, "%Y-%m-%d %H:%M:%S");
    }
    std::time_t time = std::mktime(&tm);
    return std::chrono::system_clock::from_time_t(time);
}

std::string timestampToString(const Timestamp& ts) {
    std::time_t time = std::chrono::system_clock::to_time_t(ts);
    std::tm tm = *std::localtime(&time);
    std::stringstream ss;
    ss << std::put_time(&tm, "%Y-%m-%dT%H:%M:%S");
    return ss.str();
}

std::string getTimeOfDay(const Timestamp& ts) {
    std::time_t time = std::chrono::system_clock::to_time_t(ts);
    std::tm tm = *std::localtime(&time);
    int hour = tm.tm_hour;
    
    if (hour >= 5 && hour < 12) return "morning";
    if (hour >= 12 && hour < 14) return "noon";
    if (hour >= 14 && hour < 18) return "afternoon";
    if (hour >= 18 && hour < 22) return "evening";
    return "night";
}

std::string getDayType(const Timestamp& ts) {
    std::time_t time = std::chrono::system_clock::to_time_t(ts);
    std::tm tm = *std::localtime(&time);
    int wday = tm.tm_wday;
    
    if (wday == 0 || wday == 6) return "weekend";
    return "weekday";
}

} // namespace pos
