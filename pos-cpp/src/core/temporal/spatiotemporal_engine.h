/**
 * @file spatiotemporal_engine.h
 * @brief 时空推理引擎
 * 
 * 提供RTree空间索引、时间索引和时空推理功能
 */

#pragma once

#include "../common/types.h"
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <map>
#include <set>
#include <boost/geometry.hpp>
#include <boost/geometry/index/rtree.hpp>

namespace personal_ontology {
namespace temporal {

namespace bg = boost::geometry;
namespace bgi = boost::geometry::index;

// 空间数据类型
using Point = bg::model::point<double, 2, bg::cs::cartesian>;
using Box = bg::model::box<Point>;
using SpatialEntry = std::pair<Box, MemoryId>;

/**
 * @brief 时间间隔
 */
struct TimeInterval {
    Timestamp start;
    Timestamp end;
    
    TimeInterval() : start(0), end(0) {}
    TimeInterval(Timestamp s, Timestamp e) : start(s), end(e) {}
    
    [[nodiscard]] bool contains(Timestamp t) const {
        return t >= start && t <= end;
    }
    
    [[nodiscard]] bool overlaps(const TimeInterval& other) const {
        return start <= other.end && end >= other.start;
    }
    
    [[nodiscard]] TimeInterval intersect(const TimeInterval& other) const {
        return TimeInterval(
            std::max(start, other.start),
            std::min(end, other.end)
        );
    }
    
    [[nodiscard]] int64_t duration() const {
        return end - start;
    }
};

/**
 * @brief 时空事件
 */
struct SpatiotemporalEvent {
    MemoryId memory_id;
    std::optional<GeoPoint> location;
    TimeInterval time_interval;
    std::string event_type;
    Properties properties;
    
    [[nodiscard]] bool isValid() const {
        return !memory_id.empty() && time_interval.start <= time_interval.end;
    }
};

/**
 * @brief 时间关系类型
 */
enum class TemporalRelation {
    BEFORE,         // 在...之前
    AFTER,          // 在...之后
    MEETS,          // 接触 (A结束=B开始)
    MET_BY,         // 被接触
    OVERLAPS,       // 重叠
    OVERLAPPED_BY,  // 被重叠
    DURING,         // 在...期间
    CONTAINS,       // 包含
    STARTS,         // 同时开始
    STARTED_BY,     // 被同时开始
    FINISHES,       // 同时结束
    FINISHED_BY,    // 被同时结束
    EQUALS          // 相等
};

[[nodiscard]] inline std::string temporalRelationToString(TemporalRelation rel) {
    switch (rel) {
        case TemporalRelation::BEFORE: return "before";
        case TemporalRelation::AFTER: return "after";
        case TemporalRelation::MEETS: return "meets";
        case TemporalRelation::MET_BY: return "met_by";
        case TemporalRelation::OVERLAPS: return "overlaps";
        case TemporalRelation::OVERLAPPED_BY: return "overlapped_by";
        case TemporalRelation::DURING: return "during";
        case TemporalRelation::CONTAINS: return "contains";
        case TemporalRelation::STARTS: return "starts";
        case TemporalRelation::STARTED_BY: return "started_by";
        case TemporalRelation::FINISHES: return "finishes";
        case TemporalRelation::FINISHED_BY: return "finished_by";
        case TemporalRelation::EQUALS: return "equals";
        default: return "unknown";
    }
}

/**
 * @brief 时空查询
 */
struct SpatiotemporalQuery {
    std::optional<TimeInterval> time_range;
    std::optional<GeoBoundingBox> spatial_bounds;
    std::optional<std::string> event_type_filter;
    size_t limit = 100;
    bool include_historical = true;
    bool sort_by_time = true;
};

/**
 * @brief 轨迹点
 */
struct TrajectoryPoint {
    GeoPoint location;
    Timestamp time;
    double speed;  // m/s
    double heading;  // 方向角度
    Properties attributes;
};

/**
 * @brief 轨迹
 */
struct Trajectory {
    EntityId entity_id;
    std::vector<TrajectoryPoint> points;
    
    [[nodiscard]] TimeInterval getTimeRange() const {
        if (points.empty()) return TimeInterval();
        return TimeInterval(points.front().time, points.back().time);
    }
    
    [[nodiscard]] GeoBoundingBox getBoundingBox() const {
        if (points.empty()) return GeoBoundingBox{0,0,0,0};
        
        double min_lon = points[0].location.longitude;
        double max_lon = min_lon;
        double min_lat = points[0].location.latitude;
        double max_lat = min_lat;
        
        for (const auto& point : points) {
            min_lon = std::min(min_lon, point.location.longitude);
            max_lon = std::max(max_lon, point.location.longitude);
            min_lat = std::min(min_lat, point.location.latitude);
            max_lat = std::max(max_lat, point.location.latitude);
        }
        
        return GeoBoundingBox{min_lon, min_lat, max_lon, max_lat};
    }
    
    [[nodiscard]] double getTotalDistance() const;
};

/**
 * @brief 时空推理引擎
 */
class SpatiotemporalEngine {
public:
    /**
     * @brief 构造函数
     */
    SpatiotemporalEngine();
    
    /**
     * @brief 析构函数
     */
    ~SpatiotemporalEngine();
    
    // 禁用拷贝，允许移动
    SpatiotemporalEngine(const SpatiotemporalEngine&) = delete;
    SpatiotemporalEngine& operator=(const SpatiotemporalEngine&) = delete;
    SpatiotemporalEngine(SpatiotemporalEngine&&) noexcept;
    SpatiotemporalEngine& operator=(SpatiotemporalEngine&&) noexcept;
    
    /**
     * @brief 初始化引擎
     */
    [[nodiscard]] Result<bool> initialize();
    
    /**
     * @brief 关闭引擎
     */
    void shutdown();
    
    // ========================================================================
    // 空间索引操作
    // ========================================================================
    
    /**
     * @brief 添加空间索引条目
     * @param id 记忆ID
     * @param location 位置
     * @param time 时间戳
     * @return 是否成功
     */
    [[nodiscard]] Result<bool> addSpatialIndex(
        const MemoryId& id, 
        const GeoPoint& location,
        Timestamp time);
    
    /**
     * @brief 添加空间区域索引
     * @param id 记忆ID
     * @param bbox 边界框
     * @param time_interval 时间间隔
     * @return 是否成功
     */
    [[nodiscard]] Result<bool> addSpatialRegionIndex(
        const MemoryId& id,
        const GeoBoundingBox& bbox,
        const TimeInterval& time_interval);
    
    /**
     * @brief 移除空间索引
     * @param id 记忆ID
     * @return 是否成功
     */
    [[nodiscard]] Result<bool> removeSpatialIndex(const MemoryId& id);
    
    /**
     * @brief 空间范围查询
     * @param bbox 查询边界框
     * @param limit 返回数量限制
     * @return 记忆ID列表或错误
     */
    [[nodiscard]] Result<std::vector<MemoryId>> spatialQuery(
        const GeoBoundingBox& bbox, 
        size_t limit = 100);
    
    /**
     * @brief 最近邻查询
     * @param point 查询点
     * @param k 返回数量
     * @return 带距离的记忆ID列表或错误
     */
    [[nodiscard]] Result<std::vector<std::pair<MemoryId, double>>> nearestNeighborQuery(
        const GeoPoint& point, 
        size_t k = 10);
    
    /**
     * @brief 半径查询
     * @param center 中心点
     * @param radius_meters 半径(米)
     * @return 记忆ID列表或错误
     */
    [[nodiscard]] Result<std::vector<MemoryId>> radiusQuery(
        const GeoPoint& center,
        double radius_meters);
    
    // ========================================================================
    // 时间索引操作
    // ========================================================================
    
    /**
     * @brief 添加时间索引
     * @param id 记忆ID
     * @param interval 时间间隔
     * @return 是否成功
     */
    [[nodiscard]] Result<bool> addTemporalIndex(
        const MemoryId& id,
        const TimeInterval& interval);
    
    /**
     * @brief 移除时间索引
     * @param id 记忆ID
     * @return 是否成功
     */
    [[nodiscard]] Result<bool> removeTemporalIndex(const MemoryId& id);
    
    /**
     * @brief 时间范围查询
     * @param interval 查询时间范围
     * @param limit 返回数量
     * @return 记忆ID列表或错误
     */
    [[nodiscard]] Result<std::vector<MemoryId>> temporalQuery(
        const TimeInterval& interval,
        size_t limit = 100);
    
    /**
     * @brief 获取时间最近的记忆
     * @param time 参考时间
     * @param k 返回数量
     * @return 记忆ID列表或错误
     */
    [[nodiscard]] Result<std::vector<MemoryId>> nearestTimeQuery(
        Timestamp time,
        size_t k = 10);
    
    // ========================================================================
    // 时空联合查询
    // ========================================================================
    
    /**
     * @brief 时空范围查询
     * @param query 查询条件
     * @return 记忆ID列表或错误
     */
    [[nodiscard]] Result<std::vector<MemoryId>> spatiotemporalQuery(
        const SpatiotemporalQuery& query);
    
    /**
     * @brief 轨迹查询
     * @param entity_id 实体ID
     * @param time_range 时间范围
     * @return 轨迹或错误
     */
    [[nodiscard]] Result<Trajectory> getTrajectory(
        const EntityId& entity_id,
        const TimeInterval& time_range);
    
    /**
     * @brief 轨迹相似度查询
     * @param reference_trajectory 参考轨迹
     * @param threshold 相似度阈值
     * @return 相似轨迹的实体ID列表或错误
     */
    [[nodiscard]] Result<std::vector<EntityId>> findSimilarTrajectories(
        const Trajectory& reference_trajectory,
        double threshold = 0.8);
    
    // ========================================================================
    // 时间推理
    // ========================================================================
    
    /**
     * @brief 推断两个时间间隔的关系
     * @param t1 时间间隔1
     * @param t2 时间间隔2
     * @return 时间关系
     */
    [[nodiscard]] TemporalRelation inferTemporalRelation(
        const TimeInterval& t1,
        const TimeInterval& t2) const;
    
    /**
     * @brief 推断事件顺序
     * @param event1 事件1
     * @param event2 事件2
     * @return 如果event1在event2之前返回true
     */
    [[nodiscard]] bool isBefore(const SpatiotemporalEvent& event1, 
                                const SpatiotemporalEvent& event2) const;
    
    /**
     * @brief 查找因果关系候选
     * @param event 结果事件
     * @param time_window_ms 时间窗口(毫秒)
     @return 可能的原因事件ID列表或错误
     */
    [[nodiscard]] Result<std::vector<MemoryId>> findCausalCandidates(
        const SpatiotemporalEvent& event,
        int64_t time_window_ms = 60000);  // 默认1分钟
    
    /**
     * @brief 时间聚合查询
     * @param interval 查询间隔
     * @param aggregation_ms 聚合粒度(毫秒)
     * @return 每个时间段的记忆数量
     */
    [[nodiscard]] Result<std::map<Timestamp, size_t>> temporalAggregation(
        const TimeInterval& interval,
        int64_t aggregation_ms = 3600000);  // 默认1小时
    
    // ========================================================================
    // 空间推理
    // ========================================================================
    
    /**
     * @brief 计算两点间的距离
     * @param p1 点1
     * @param p2 点2
     * @return 距离(米)
     */
    [[nodiscard]] static double distance(const GeoPoint& p1, const GeoPoint& p2);
    
    /**
     * @brief 计算点到线段的距离
     * @param point 点
     * @param line_start 线段起点
     * @param line_end 线段终点
     * @return 距离(米)
     */
    [[nodiscard]] static double distanceToLine(
        const GeoPoint& point,
        const GeoPoint& line_start,
        const GeoPoint& line_end);
    
    /**
     * @brief 计算地理边界框的面积
     * @param bbox 边界框
     * @return 面积(平方米)
     */
    [[nodiscard]] static double area(const GeoBoundingBox& bbox);
    
    /**
     * @brief 判断点是否在多边形内
     * @param point 点
     * @param polygon 多边形顶点
     * @return 是否在内部
     */
    [[nodiscard]] static bool pointInPolygon(
        const GeoPoint& point,
        const std::vector<GeoPoint>& polygon);
    
    /**
     * @brief 查找共位事件
     * @param location 位置
     * @param radius_meters 半径
     * @param time_window_ms 时间窗口
     * @return 共位事件ID列表或错误
     */
    [[nodiscard]] Result<std::vector<MemoryId>> findCoLocatedEvents(
        const GeoPoint& location,
        double radius_meters,
        int64_t time_window_ms);
    
    /**
     * @brief 计算轨迹相似度 (使用DTW算法)
     * @param t1 轨迹1
     * @param t2 轨迹2
     * @return 相似度分数(0-1)
     */
    [[nodiscard]] static double trajectorySimilarityDTW(
        const Trajectory& t1,
        const Trajectory& t2);
    
    // ========================================================================
    // 统计和监控
    // ========================================================================
    
    struct Statistics {
        size_t spatial_index_size = 0;
        size_t temporal_index_size = 0;
        size_t event_count = 0;
        GeoBoundingBox spatial_bounds;
        TimeInterval temporal_range;
    };
    
    [[nodiscard]] Statistics getStatistics() const;
    
    /**
     * @brief 重建索引
     */
    [[nodiscard]] Result<bool> rebuildIndices();
    
private:
    // RTree空间索引
    using RTree = bgi::rtree<SpatialEntry, bgi::rstar<16>>;
    std::unique_ptr<RTree> spatial_index_;
    mutable std::shared_mutex spatial_mutex_;
    
    // 时间索引 (时间->记忆ID集合)
    std::map<Timestamp, std::set<MemoryId>> temporal_index_;
    mutable std::shared_mutex temporal_mutex_;
    
    // 记忆ID到时空信息的映射
    struct SpatiotemporalInfo {
        std::optional<Box> spatial_bounds;
        TimeInterval time_interval;
    };
    std::unordered_map<MemoryId, SpatiotemporalInfo> st_info_;
    mutable std::shared_mutex info_mutex_;
    
    // 轨迹存储 (实体ID -> 轨迹)
    std::unordered_map<EntityId, Trajectory> trajectories_;
    mutable std::shared_mutex trajectory_mutex_;
    
    bool initialized_ = false;
    
    // 辅助函数
    static Point geoToPoint(const GeoPoint& geo);
    static Box bboxToBox(const GeoBoundingBox& bbox);
};

} // namespace temporal
} // namespace personal_ontology
