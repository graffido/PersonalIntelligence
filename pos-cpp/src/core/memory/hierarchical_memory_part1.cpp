/**
 * @file hierarchical_memory.cpp (Part 1)
 * @brief 四层分层记忆存储系统实现 - 构造函数与初始化
 */

#include "hierarchical_memory.h"
#include <hnswlib/hnswlib.h>
#include <rocksdb/write_batch.h>
#include <rocksdb/snapshot.h>
#include <folly/json.h>
#include <chrono>
#include <fstream>
#include <filesystem>
#include <math>

namespace personal_ontology {
namespace memory {

// =============================================================================
// 构造函数和析构函数
// =============================================================================

QuadLayerMemoryStore::QuadLayerMemoryStore(const HierarchicalMemoryConfig& config)
    : config_(config) {
    // 初始化各层访问计数
    for (auto& count : hourly_access_count_) {
        count = 0;
    }
}

QuadLayerMemoryStore::~QuadLayerMemoryStore() {
    if (initialized_) {
        shutdown();
    }
}

QuadLayerMemoryStore::QuadLayerMemoryStore(QuadLayerMemoryStore&& other) noexcept
    : config_(std::move(other.config_))
    , initialized_(other.initialized_.load())
    , sensory_buffer_(std::move(other.sensory_buffer_))
    , attention_focus_(std::move(other.attention_focus_))
    , working_memory_(std::move(other.working_memory_))
    , focused_id_(std::move(other.focused_id_))
    , chunks_(std::move(other.chunks_))
    , ltm_db_(std::move(other.ltm_db_))
    , ltm_cf_memories_(other.ltm_cf_memories_)
    , ltm_cf_embeddings_(other.ltm_cf_embeddings_)
    , ltm_cf_indices_(other.ltm_cf_indices_)
    , ltm_cf_relations_(other.ltm_cf_relations_)
    , ltm_cache_(std::move(other.ltm_cache_))
    , ltm_lru_list_(std::move(other.ltm_lru_list_))
    , hnsw_space_(std::move(other.hnsw_space_))
    , hnsw_index_(std::move(other.hnsw_index_))
    , spatial_index_(std::move(other.spatial_index_))
    , temporal_index_(std::move(other.temporal_index_))
    , param_db_(std::move(other.param_db_))
    , user_prefs_(std::move(other.user_prefs_))
    , access_pattern_(std::move(other.access_pattern_))
    , parameters_(std::move(other.parameters_))
    , access_history_(std::move(other.access_history_))
    , cache_hits_(other.cache_hits_.load())
    , cache_misses_(other.cache_misses_.load())
    , total_access_time_ms_(other.total_access_time_ms_.load())
    , access_count_(other.access_count_.load()) {
    
    other.initialized_ = false;
    other.ltm_cf_memories_ = nullptr;
    other.ltm_cf_embeddings_ = nullptr;
    other.ltm_cf_indices_ = nullptr;
    other.ltm_cf_relations_ = nullptr;
}

QuadLayerMemoryStore& QuadLayerMemoryStore::operator=(QuadLayerMemoryStore&& other) noexcept {
    if (this != &other) {
        if (initialized_) {
            shutdown();
        }
        
        config_ = std::move(other.config_);
        initialized_ = other.initialized_.load();
        sensory_buffer_ = std::move(other.sensory_buffer_);
        attention_focus_ = std::move(other.attention_focus_);
        working_memory_ = std::move(other.working_memory_);
        focused_id_ = std::move(other.focused_id_);
        chunks_ = std::move(other.chunks_);
        ltm_db_ = std::move(other.ltm_db_);
        ltm_cf_memories_ = other.ltm_cf_memories_;
        ltm_cf_embeddings_ = other.ltm_cf_embeddings_;
        ltm_cf_indices_ = other.ltm_cf_indices_;
        ltm_cf_relations_ = other.ltm_cf_relations_;
        ltm_cache_ = std::move(other.ltm_cache_);
        ltm_lru_list_ = std::move(other.ltm_lru_list_);
        hnsw_space_ = std::move(other.hnsw_space_);
        hnsw_index_ = std::move(other.hnsw_index_);
        spatial_index_ = std::move(other.spatial_index_);
        temporal_index_ = std::move(other.temporal_index_);
        param_db_ = std::move(other.param_db_);
        user_prefs_ = std::move(other.user_prefs_);
        access_pattern_ = std::move(other.access_pattern_);
        parameters_ = std::move(other.parameters_);
        access_history_ = std::move(other.access_history_);
        cache_hits_ = other.cache_hits_.load();
        cache_misses_ = other.cache_misses_.load();
        total_access_time_ms_ = other.total_access_time_ms_.load();
        access_count_ = other.access_count_.load();
        
        other.initialized_ = false;
        other.ltm_cf_memories_ = nullptr;
        other.ltm_cf_embeddings_ = nullptr;
        other.ltm_cf_indices_ = nullptr;
        other.ltm_cf_relations_ = nullptr;
    }
    return *this;
}

// =============================================================================
// 初始化
// =============================================================================

Result<bool> QuadLayerMemoryStore::initialize() {
    try {
        // 初始化各层
        auto sensory_result = initializeSensoryLayer();
        if (sensory_result.isError()) return sensory_result;
        
        auto working_result = initializeWorkingLayer();
        if (working_result.isError()) return working_result;
        
        auto ltm_result = initializeLongTermLayer();
        if (ltm_result.isError()) return ltm_result;
        
        auto param_result = initializeParameterLayer();
        if (param_result.isError()) return param_result;
        
        // 启动后台维护线程
        if (config_.enable_auto_consolidation) {
            stop_maintenance_ = false;
            maintenance_thread_ = std::make_unique<std::thread>(
                &QuadLayerMemoryStore::maintenanceLoop, this);
        }
        
        initialized_ = true;
        return Result<bool>(true);
        
    } catch (const std::exception& e) {
        return Result<bool>(ErrorCode::INTERNAL_ERROR,
            std::format("Initialization failed: {}", e.what()));
    }
}

Result<bool> QuadLayerMemoryStore::initializeSensoryLayer() {
    // 预分配感知缓冲区
    sensory_buffer_.reserve(config_.sensory.buffer_size);
    return Result<bool>(true);
}

Result<bool> QuadLayerMemoryStore::initializeWorkingLayer() {
    // 工作记忆使用纯内存，无需特殊初始化
    working_memory_.reserve(config_.working.capacity * 2);
    return Result<bool>(true);
}

Result<bool> QuadLayerMemoryStore::initializeLongTermLayer() {
    try {
        // 创建数据目录
        std::filesystem::create_directories(config_.long_term.storage_path);
        std::filesystem::create_directories(config_.long_term.hnsw_index_path);
        
        // 初始化空间索引
        spatial_index_ = std::make_unique<RTree>();
        
        // 初始化HNSW向量索引
        hnsw_space_ = std::make_unique<hnswlib::L2Space>(config_.long_term.vector_dim);
        hnsw_index_ = std::make_unique<hnswlib::HierarchicalNSW<float>>(
            hnsw_space_.get(),
            10000,  // 初始容量
            config_.long_term.hnsw_m,
            config_.long_term.hnsw_ef_construction
        );
        hnsw_index_>setEf(config_.long_term.hnsw_ef_search);
        
        // 从磁盘加载HNSW索引（如果存在）
        std::string hnsw_file = config_.long_term.hnsw_index_path + "/hnsw.index";
        if (std::filesystem::exists(hnsw_file)) {
            hnsw_index_>loadIndex(hnsw_file);
        }
        
        // 初始化RocksDB
        rocksdb::Options options;
        options.create_if_missing = true;
        options.max_open_files = 1000;
        options.write_buffer_size = 128 * 1024 * 1024;  // 128MB
        options.target_file_size_base = 128 * 1024 * 1024;
        
        if (config_.long_term.enable_compression) {
            options.compression = rocksdb::kLZ4Compression;
            options.bottommost_compression = rocksdb::kZSTD;
        }
        
        std::vector<rocksdb::ColumnFamilyDescriptor> cfs;
        cfs.push_back(rocksdb::ColumnFamilyDescriptor("memories", rocksdb::ColumnFamilyOptions()));
        cfs.push_back(rocksdb::ColumnFamilyDescriptor("embeddings", rocksdb::ColumnFamilyOptions()));
        cfs.push_back(rocksdb::ColumnFamilyDescriptor("indices", rocksdb::ColumnFamilyOptions()));
        cfs.push_back(rocksdb::ColumnFamilyDescriptor("relations", rocksdb::ColumnFamilyOptions()));
        
        std::vector<rocksdb::ColumnFamilyHandle*> handles;
        rocksdb::Status status = rocksdb::DB::Open(
            options, config_.long_term.storage_path, cfs, &handles, &ltm_db_);
        
        if (!status.ok()) {
            return Result<bool>(ErrorCode::STORAGE_WRITE_ERROR,
                std::format("Failed to open LTM database: {}", status.ToString()));
        }
        
        ltm_cf_memories_ = handles[0];
        ltm_cf_embeddings_ = handles[1];
        ltm_cf_indices_ = handles[2];
        ltm_cf_relations_ = handles[3];
        
        // 预分配LRU缓存
        ltm_cache_.reserve(config_.long_term.lru_cache_size);
        
        return Result<bool>(true);
        
    } catch (const std::exception& e) {
        return Result<bool>(ErrorCode::INTERNAL_ERROR,
            std::format("LTM initialization failed: {}", e.what()));
    }
}

Result<bool> QuadLayerMemoryStore::initializeParameterLayer() {
    try {
        std::filesystem::create_directories(config_.parameter.storage_path);
        
        rocksdb::Options options;
        options.create_if_missing = true;
        options.max_open_files = 100;
        
        rocksdb::Status status = rocksdb::DB::Open(
            options, config_.parameter.storage_path, &param_db_);
        
        if (!status.ok()) {
            return Result<bool>(ErrorCode::STORAGE_WRITE_ERROR,
                std::format("Failed to open parameter database: {}", status.ToString()));
        }
        
        // 从磁盘加载现有参数
        std::unique_ptr<rocksdb::Iterator> it(param_db_>NewIterator(rocksdb::ReadOptions()));
        for (it->SeekToFirst(); it->Valid(); it->Next()) {
            std::string key = it->key().ToString();
            std::string value = it->value().ToString();
            
            // 解析值向量
            std::vector<float> values;
            const float* data = reinterpret_cast<const float*>(value.data());
            size_t count = value.size() / sizeof(float);
            values.assign(data, data + count);
            
            parameters_[key] = ParameterMemoryEntry(key, values);
        }
        
        return Result<bool>(true);
        
    } catch (const std::exception& e) {
        return Result<bool>(ErrorCode::INTERNAL_ERROR,
            std::format("Parameter layer initialization failed: {}", e.what()));
    }
}

// =============================================================================
// 关闭
// =============================================================================

void QuadLayerMemoryStore::shutdown() {
    if (!initialized_) return;
    
    // 停止维护线程
    if (maintenance_thread_ && maintenance_thread_>joinable()) {
        stop_maintenance_ = true;
        maintenance_cv_.notify_all();
        maintenance_thread_>join();
    }
    
    // 保存HNSW索引
    if (hnsw_index_) {
        std::string hnsw_file = config_.long_term.hnsw_index_path + "/hnsw.index";
        hnsw_index_>saveIndex(hnsw_file);
    }
    
    // 保存参数
    for (const auto& [key, entry] : parameters_) {
        std::string value(reinterpret_cast<const char*>(entry.values.data()),
                         entry.values.size() * sizeof(float));
        param_db_>Put(rocksdb::WriteOptions(), key, value);
    }
    
    // 关闭RocksDB
    if (ltm_db_) {
        delete ltm_cf_memories_;
        delete ltm_cf_embeddings_;
        delete ltm_cf_indices_;
        delete ltm_cf_relations_;
        ltm_cf_memories_ = nullptr;
        ltm_cf_embeddings_ = nullptr;
        ltm_cf_indices_ = nullptr;
        ltm_cf_relations_ = nullptr;
        ltm_db_.reset();
    }
    
    if (param_db_) {
        param_db_.reset();
    }
    
    initialized_ = false;
}

} // namespace memory
} // namespace personal_ontology
