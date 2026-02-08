#pragma once

#include "core/common/types.h"
#include <string>
#include <vector>
#include <map>

namespace pos {

// 时空查询结构
struct SpatiotemporalQuery {
    std::optional<Timestamp> time_start;
    std::optional<Timestamp> time_end;
    std::optional<GeoPoint> location;
    std::optional<double> radius_meters;
    std::vector<std::string> entity_types;
    std::string temporal_relation{"at"};  // at/before/after/during/nearby
    
    // 高级选项
    bool include_nearby_times{false};
    double temporal_tolerance_hours{1.0};
};

// 时空索引
class SpatiotemporalIndex {
public:
    explicit SpatiotemporalIndex(const std::string& storage_path);
    ~SpatiotemporalIndex();
    
    // 索引操作
    bool index(const MemoryId& memory, 
               const Timestamp& time,
               const std::optional<GeoPoint>& location);
    
    bool remove(const MemoryId& memory);
    
    // 查询
    std::vector<MemoryId> query(const SpatiotemporalQuery& query) const;
    
    // 邻近查询
    std::vector<MemoryId> queryNearby(
        const GeoPoint& center,
        double radius_meters,
        const std::optional<Timestamp>& time_hint = std::nullopt
    ) const;
    
    // 时间范围查询
    std::vector<MemoryId> queryTimeRange(
        const Timestamp::duration& duration_before,
        const Timestamp::duration& duration_after,
        const Timestamp& reference_time
    ) const;
    
    // 时空模式发现
    struct TemporalPattern {
        std::string pattern_type;  // daily, weekly, monthly
        std::string description;
        int frequency;
        float confidence;
        std::vector<MemoryId> examples;
    };
    
    std::vector<TemporalPattern> discoverPatterns(
        const std::string& pattern_type = "all",
        int min_occurrences = 3
    ) const;

private:
    class Impl;
    std::unique_ptr<Impl> pimpl_;
};

// 时空推理引擎
class SpatiotemporalReasoner {
public:
    struct SituationContext {
        // 时间情境
        std::string time_of_day;      // morning/afternoon/evening/night
        std::string day_type;         // weekday/weekend/holiday
        std::string temporal_phase;   // work/leisure/commute/rest/sleep
        
        // 空间情境
        std::optional<GeoPoint> current_location;
        std::string location_context; // home/office/outdoor/transit/commercial
        std::vector<std::string> nearby_pois;
        
        // 综合情境
        std::string activity_context; // working/eating/socializing/relaxing/commuting
        float routine_probability{0.5};  // 当前行为符合日常规律的概率
    };
    
    explicit SpatiotemporalReasoner(SpatiotemporalIndex& index);
    
    // 推理当前情境
    SituationContext inferCurrentSituation(
        const std::optional<Timestamp>& explicit_time = std::nullopt,
        const std::optional<GeoPoint>& explicit_location = std::nullopt
    ) const;
    
    // 可达性分析
    struct ReachabilityResult {
        bool reachable;
        double distance_meters;
        Duration estimated_duration;
        std::string transport_mode;
        std::vector<GeoPoint> route;
    };
    
    ReachabilityResult analyzeReachability(
        const GeoPoint& from,
        const GeoPoint& to,
        const Timestamp& deadline,
        const std::string& transport_mode = "driving"
    ) const;
    
    // 预测下一个可能位置
    std::vector<std::pair<GeoPoint, float>> predictNextLocations(
        const Timestamp& current_time,
        const GeoPoint& current_location,
        int top_k = 3
    ) const;
    
    // 时间冲突检测
    bool hasTimeConflict(
        const Timestamp& proposed_start,
        const Timestamp& proposed_end,
        const std::optional<GeoPoint>& proposed_location = std::nullopt
    ) const;

private:
    SpatiotemporalIndex& index_;
};

// 外部地理服务客户端
class GeospatialServiceClient {
public:
    struct Config {
        std::string provider{"nominatim"};  // nominatim/google/mapbox
        std::string endpoint;
        std::string api_key;
        float rate_limit{1.0};  // requests per second
    };
    
    explicit GeospatialServiceClient(const Config& config);
    ~GeospatialServiceClient();
    
    // 地理编码
    struct GeocodeResult {
        GeoPoint point;
        std::string formatted_address;
        std::string place_type;
        std::unordered_map<std::string, std::string> components;
        std::string timezone;
    };
    
    std::optional<GeocodeResult> geocode(const std::string& address);
    std::optional<std::string> reverseGeocode(const GeoPoint& point);
    
    // POI搜索
    struct POI {
        std::string name;
        std::string category;
        GeoPoint location;
        float rating{0.0};
        std::unordered_map<std::string, std::string> attributes;
    };
    
    std::vector<POI> searchNearby(
        const GeoPoint& center,
        double radius_meters,
        const std::string& category = ""
    );
    
    // 路线规划
    struct Route {
        double distance_meters;
        Duration duration;
        std::vector<GeoPoint> waypoints;
        std::string polyline;
        std::string traffic_condition;
    };
    
    std::optional<Route> getRoute(
        const GeoPoint& from,
        const GeoPoint& to,
        const std::string& mode = "driving"  // driving/walking/transit/cycling
    );
    
    // 时区查询
    std::string getTimezone(const GeoPoint& point);

private:
    Config config_;
    class Impl;
    std::unique_ptr<Impl> pimpl_;
};

} // namespace pos
