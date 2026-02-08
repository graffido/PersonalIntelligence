/**
 * @file memory_store.cpp
 * @brief 分层记忆存储系统实现
 * 
 * 实现三层存储架构和索引系统
 */

#include "memory_store.h"
#include <hnswlib/hnswlib.h>
#include <rocksdb/write_batch.h>
#include <rocksdb/snapshot.h>
#include <folly/json.h>
#include <folly/executors/CPUThreadPoolExecutor.h>
#include <chrono>
#include <fstream>
#include <filesystem>

namespace personal_ontology {
namespace memory {

// =============================================================================
// 构造函数和析构函数
// =============================================================================

HierarchicalMemoryStore::HierarchicalMemoryStore(const StorageLayerConfig& config)
    : config_(config) {
    // 空间索引在initialize中创建
}

HierarchicalMemoryStore::~HierarchicalMemoryStore() {
    if (initialized_) {
        shutdown();
    }
}

HierarchicalMemoryStore::HierarchicalMemoryStore(HierarchicalMemoryStore&& other) noexcept
    : config_(std::move(other.config_))
    , initialized_(other.initialized_.load())
    , working_memory_(std::move(other.working_memory_))
    , wm_lru_queue_(std::move(other.wm_lru_queue_))
    , short_term_cache_(std::move(other.short_term_cache_))
    , stm_db_(std::move(other.stm_db_))
    , stm_cf_default_(other.stm_cf_default_)
    , stm_cf_metadata_(other.stm_cf_metadata_)
    , ltm_db_(std::move(other.ltm_db_))
    , ltm_cf_memories_(other.ltm_cf_memories_)
    , ltm_cf_embeddings_(other.ltm_cf_embeddings_)
    , ltm_cf_indices_(other.ltm_cf_indices_)
    , hnsw_space_(std::move(other.hnsw_space_))
    , hnsw_index_(std::move(other.hnsw_index_))
    , spatial_index_(std::move(other.spatial_index_))
    , temporal_index_(std::move(other.temporal_index_))
    , cache_hits_(other.cache_hits_.load())
    , cache_misses_(other.cache_misses_.load())
    , total_access_time_ms_(other.total_access_time_ms_.load())
    , access_count_(other.access_count_.load()) {
    other.initialized_ = false;
    other.stm_cf_default_ = nullptr;
    other.stm_cf_metadata_ = nullptr;
    other.ltm_cf_memories_ = nullptr;
    other.ltm_cf_embeddings_ = nullptr;
    other.ltm_cf_indices_ = nullptr;
}

HierarchicalMemoryStore& HierarchicalMemoryStore::operator=(HierarchicalMemoryStore&& other) noexcept {
    if (this != &other) {
        if (initialized_) {
            shutdown();
        }
        
        config_ = std::move(other.config_);
        initialized_ = other.initialized_.load();
        working_memory_ = std::move(other.working_memory_);
        wm_lru_queue_ = std::move(other.wm_lru_queue_);
        short_term_cache_ = std::move(other.short_term_cache_);
        stm_db_ = std::move(other.stm_db_);
        stm_cf_default_ = other.stm_cf_default_;
        stm_cf_metadata_ = other.stm_cf_metadata_;
        ltm_db_ = std::move(other.ltm_db_);
        ltm_cf_memories_ = other.ltm_cf_memories_;
        ltm_cf_embeddings_ = other.ltm_cf_embeddings_;
        ltm_cf_indices_ = other.ltm_cf_indices_;
        hnsw_space_ = std::move(other.hnsw_space_);
        hnsw_index_ = std::move(other.hnsw_index_);
        spatial_index_ = std::move(other.spatial_index_);
        temporal_index_ = std::move(other.temporal_index_);
        cache_hits_ = other.cache_hits_.load();
        cache_misses_ = other.cache_misses_.load();
        total_access_time_ms_ = other.total_access_time_ms_.load();
        access_count_ = other.access_count_.load();
        
        other.initialized_ = false;
        other.stm_cf_default_ = nullptr;
        other.stm_cf_metadata_ = nullptr;
        other.ltm_cf_memories_ = nullptr;
        other.ltm_cf_embeddings_ = nullptr;
        other.ltm_cf_indices_ = nullptr;
    }
    return *this;
}

// =============================================================================
// 初始化和关闭
// =============================================================================

Result<bool> HierarchicalMemoryStore::initialize() {
    try {
        // 创建数据目录
        std::filesystem::create_directories(config_.short_term_path);
        std::filesystem::create_directories(config_.long_term_path);
        std::filesystem::create_directories(config_.hnsw_index_path);
        
        // 初始化空间索引
        spatial_index_ = std::make_unique<RTree>();
        
        // 初始化HNSW向量索引
        hnsw_space_ = std::make_unique<hnswlib::L2Space>(config_.vector_dim);
        hnsw_index_ = std::make_unique<hnswlib::HierarchicalNSW<float>>(
            hnsw_space_.get(),
            10000,  // 初始容量，会自动扩展
            config_.hnsw_m,
            config_.hnsw_ef_construction
        );
        hnsw_index_>-setEf(config_.hnsw_ef_search);
        
        // 从磁盘加载HNSW索引（如果存在）
        std::string hnsw_file = config_.hnsw_index_path + "/hnsw.index";
        if (std::filesystem::exists(hnsw_file)) {
            hnsw_index_>-loadIndex(hnsw_file);
        }
        
        // 初始化短期记忆RocksDB
        rocksdb::Options stm_options;
        stm_options.create_if_missing = true;
        stm_options.max_open_files = 500;
        stm_options.write_buffer_size = 64 * 1024 * 1024;  // 64MB
        stm_options.target_file_size_base = 64 * 1024 * 1024;
        
        std::vector<rocksdb::ColumnFamilyDescriptor> stm_cfs;
        stm_cfs.push_back(rocksdb::ColumnFamilyDescriptor(
            rocksdb::kDefaultColumnFamilyName, rocksdb::ColumnFamilyOptions()));
        stm_cfs.push_back(rocksdb::ColumnFamilyDescriptor(
            "metadata", rocksdb::ColumnFamilyOptions()));
        
        std::vector<rocksdb::ColumnFamilyHandle*> stm_handles;
        rocksdb::Status stm_status = rocksdb::DB::Open(
            stm_options, config_.short_term_path, stm_cfs, &stm_handles, &stm_db_);
        
        if (!stm_status.ok()) {
            return Result<bool>(ErrorCode::STORAGE_WRITE_ERROR,
                std::format("Failed to open STM database: {}", stm_status.ToString()));
        }
        
        stm_cf_default_ = stm_handles[0];
        stm_cf_metadata_ = stm_handles[1];
        
        // 初始化长期记忆RocksDB
        rocksdb::Options ltm_options;
        ltm_options.create_if_missing = true;
        ltm_options.max_open_files = 1000;
        ltm_options.write_buffer_size = 128 * 1024 * 1024;  // 128MB
        ltm_options.target_file_size_base = 128 * 1024 * 1024;
        
        // 启用压缩
        ltm_options.compression = rocksdb::kLZ4Compression;
        ltm_options.bottommost_compression = rocksdb::kZSTD;
        
        std::vector<rocksdb::ColumnFamilyDescriptor> ltm_cfs;
        ltm_cfs.push_back(rocksdb::ColumnFamilyDescriptor(
            "memories", rocksdb::ColumnFamilyOptions()));
        ltm_cfs.push_back(rocksdb::ColumnFamilyDescriptor(
            "embeddings", rocksdb::ColumnFamilyOptions()));
        ltm_cfs.push_back(rocksdb::ColumnFamilyDescriptor(
            "indices", rocksdb::ColumnFamilyOptions()));
        
        std::vector<rocksdb::ColumnFamilyHandle*> ltm_handles;
        rocksdb::Status ltm_status = rocksdb::DB::Open(
            ltm_options, config_.long_term_path, ltm_cfs, &ltm_handles, &ltm_db_);
        
        if (!ltm_status.ok()) {
            return Result<bool>(ErrorCode::STORAGE_WRITE_ERROR,
                std::format("Failed to open LTM database: {}", ltm_status.ToString()));
        }
        
        ltm_cf_memories_ = ltm_handles[0];
        ltm_cf_embeddings_ = ltm_handles[1];
        ltm_cf_indices_ = ltm_handles[2];
        
        // 从长期记忆加载时空索引
        loadSpatialTemporalIndices();
        
        initialized_ = true;
        return Result<bool>(true);
        
    } catch (const std::exception& e) {
        return Result<bool>(ErrorCode::INTERNAL_ERROR,
            std::format("Initialization failed: {}", e.what()));
    }
}

void HierarchicalMemoryStore::shutdown() {
    if (!initialized_) return;
    
    // 保存HNSW索引
    if (hnsw_index_) {
        std::string hnsw_file = config_.hnsw_index_path + "/hnsw.index";
        hnsw_index_>-saveIndex(hnsw_file);
    }
    
    // 关闭RocksDB
    if (stm_db_) {
        delete stm_cf_default_;
        delete stm_cf_metadata_;
        stm_cf_default_ = nullptr;
        stm_cf_metadata_ = nullptr;
        stm_db_.reset();
    }
    
    if (ltm_db_) {
        delete ltm_cf_memories_;
        delete ltm_cf_embeddings_;
        delete ltm_cf_indices_;
        ltm_cf_memories_ = nullptr;
        ltm_cf_embeddings_ = nullptr;
        ltm_cf_indices_ = nullptr;
        ltm_db_.reset();
    }
    
    initialized_ = false;
}

// =============================================================================
// 核心CRUD操作
// =============================================================================

Result<MemoryId> HierarchicalMemoryStore::store(MemoryTrace memory, uint8_t target_layer) {
    auto start = std::chrono::high_resolution_clock::now();
    
    // 生成ID（如果未提供）
    if (memory.id.empty()) {
        memory.id = generateUUID();
    }
    memory.created_at = MemoryTrace::now();
    memory.updated_at = memory.created_at;
    
    MemoryEntry entry(memory, target_layer);
    
    try {
        switch (target_layer) {
            case 0: {  // 工作记忆
                std::unique_lock<std::shared_mutex> lock(wm_mutex_);
                
                // 检查容量并驱逐
                if (working_memory_.size() >= config_.working_memory_capacity) {
                    evictFromWorkingMemory();
                }
                
                working_memory_[memory.id] = entry;
                wm_lru_queue_.push(memory.id);
                
                // 更新索引
                if (memory.embedding.has_value()) {
                    addToHnswIndex(memory.id, memory.embedding.value());
                }
                if (memory.location.has_value()) {
                    addToSpatialIndex(memory.id, memory.location.value());
                }
                if (memory.event_time.has_value()) {
                    addToTemporalIndex(memory.id, memory.event_time.value());
                }
                break;
            }
            
            case 1: {  // 短期记忆
                std::unique_lock<std::shared_mutex> lock(stm_mutex_);
                
                auto result = writeToStm(entry);
                if (result.isError()) {
                    return Result<MemoryId>(result.errorCode(), result.errorMessage());
                }
                
                // 添加到缓存
                if (short_term_cache_.size() < config_.short_term_capacity / 2) {
                    short_term_cache_[memory.id] = entry;
                }
                break;
            }
            
            case 2: {  // 长期记忆
                std::unique_lock<std::shared_mutex> lock(ltm_mutex_);
                
                auto result = writeToLtm(entry);
                if (result.isError()) {
                    return Result<MemoryId>(result.errorCode(), result.errorMessage());
                }
                break;
            }
            
            default:
                return Result<MemoryId>(ErrorCode::INVALID_ARGUMENT,
                    std::format("Invalid target layer: {}", target_layer));
        }
        
        return Result<MemoryId>(memory.id);
        
    } catch (const std::exception& e) {
        return Result<MemoryId>(ErrorCode::STORAGE_WRITE_ERROR,
            std::format("Store failed: {}", e.what()));
    }
}

Result<std::vector<MemoryId>> HierarchicalMemoryStore::storeBatch(
    std::vector<MemoryTrace> memories, uint8_t target_layer) {
    
    std::vector<MemoryId> ids;
    ids.reserve(memories.size());
    
    // 批量写入短期或长期记忆时使用WriteBatch
    if (target_layer == 1 && stm_db_) {
        rocksdb::WriteBatch batch;
        std::unique_lock<std::shared_mutex> lock(stm_mutex_);
        
        for (auto& memory : memories) {
            if (memory.id.empty()) {
                memory.id = generateUUID();
            }
            memory.created_at = MemoryTrace::now();
            memory.updated_at = memory.created_at;
            
            MemoryEntry entry(memory, target_layer);
            std::string data = serializeMemory(memory);
            batch.Put(stm_cf_default_, memory.id, data);
            
            std::string stats_data = serializeAccessStats(entry.stats);
            batch.Put(stm_cf_metadata_, memory.id + ":stats", stats_data);
            
            ids.push_back(memory.id);
        }
        
        rocksdb::WriteOptions write_options;
        auto status = stm_db_>Write(write_options, &batch);
        if (!status.ok()) {
            return Result<std::vector<MemoryId>>(ErrorCode::STORAGE_WRITE_ERROR,
                std::format("Batch write failed: {}", status.ToString()));
        }
        
    } else if (target_layer == 2 && ltm_db_) {
        rocksdb::WriteBatch batch;
        std::unique_lock<std::shared_mutex> lock(ltm_mutex_);
        
        for (auto& memory : memories) {
            if (memory.id.empty()) {
                memory.id = generateUUID();
            }
            memory.created_at = MemoryTrace::now();
            memory.updated_at = memory.created_at;
            
            std::string data = serializeMemory(memory);
            batch.Put(ltm_cf_memories_, memory.id, data);
            
            // 存储向量嵌入
            if (memory.embedding.has_value()) {
                std::string emb_data(reinterpret_cast<const char*>(
                    memory.embedding.value().data()),
                    memory.embedding.value().size() * sizeof(float));
                batch.Put(ltm_cf_embeddings_, memory.id, emb_data);
            }
            
            ids.push_back(memory.id);
        }
        
        rocksdb::WriteOptions write_options;
        auto status = ltm_db_>Write(write_options, &batch);
        if (!status.ok()) {
            return Result<std::vector<MemoryId>>(ErrorCode::STORAGE_WRITE_ERROR,
                std::format("Batch write failed: {}", status.ToString()));
        }
        
    } else {
        // 逐个存储
        for (auto& memory : memories) {
            auto result = store(std::move(memory), target_layer);
            if (result.isError()) {
                return Result<std::vector<MemoryId>>(result.errorCode(), result.errorMessage());
            }
            ids.push_back(result.value());
        }
    }
    
    return Result<std::vector<MemoryId>>(std::move(ids));
}

Result<MemoryTrace> HierarchicalMemoryStore::retrieve(const MemoryId& id) {
    auto start = std::chrono::high_resolution_clock::now();
    
    // 1. 先查工作记忆
    {
        std::shared_lock<std::shared_mutex> lock(wm_mutex_);
        auto it = working_memory_.find(id);
        if (it != working_memory_.end()) {
            it->second.stats.recordAccess();
            wm_lru_queue_.push(id);  // 更新LRU
            cache_hits_++;
            return Result<MemoryTrace>(it->second.memory);
        }
    }
    
    // 2. 查短期记忆缓存
    {
        std::shared_lock<std::shared_mutex> lock(stm_mutex_);
        auto it = short_term_cache_.find(id);
        if (it != short_term_cache_.end()) {
            it->second.stats.recordAccess();
            cache_hits_++;
            
            // 异步更新访问统计到磁盘
            writeToStm(it->second);
            return Result<MemoryTrace>(it->second.memory);
        }
    }
    
    // 3. 查短期记忆磁盘
    {
        auto result = readFromStm(id);
        if (result.isOk() && result.value().has_value()) {
            auto entry = result.value().value();
            entry.stats.recordAccess();
            cache_hits_++;
            
            // 加载到缓存
            {
                std::unique_lock<std::shared_mutex> lock(stm_mutex_);
                if (short_term_cache_.size() >= config_.short_term_capacity / 2) {
                    evictFromShortTermCache();
                }
                short_term_cache_[id] = entry;
            }
            
            return Result<MemoryTrace>(entry.memory);
        }
    }
    
    // 4. 查长期记忆
    {
        auto result = readFromLtm(id);
        if (result.isOk() && result.value().has_value()) {
            cache_hits_++;
            return Result<MemoryTrace>(result.value().value());
        }
    }
    
    cache_misses_++;
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
    total_access_time_ms_ += duration;
    access_count_++;
    
    return Result<MemoryTrace>(ErrorCode::STORAGE_NOT_FOUND,
        std::format("Memory not found: {}", id));
}

Result<std::vector<MemoryTrace>> HierarchicalMemoryStore::retrieveBatch(
    const std::vector<MemoryId>& ids) {
    
    std::vector<MemoryTrace> results;
    results.reserve(ids.size());
    
    for (const auto& id : ids) {
        auto result = retrieve(id);
        if (result.isOk()) {
            results.push_back(std::move(result.value()));
        }
        // 忽略错误，继续获取其他记忆
    }
    
    return Result<std::vector<MemoryTrace>>(std::move(results));
}

Result<bool> HierarchicalMemoryStore::update(
    const MemoryId& id,
    std::function<void(MemoryTrace&)> update_fn) {
    
    // 尝试在各层中找到并更新记忆
    
    // 1. 工作记忆
    {
        std::unique_lock<std::shared_mutex> lock(wm_mutex_);
        auto it = working_memory_.find(id);
        if (it != working_memory_.end()) {
            update_fn(it->second.memory);
            it->second.memory.updated_at = MemoryTrace::now();
            it->second.stats.recordAccess();
            return Result<bool>(true);
        }
    }
    
    // 2. 短期记忆缓存
    {
        std::unique_lock<std::shared_mutex> lock(stm_mutex_);
        auto it = short_term_cache_.find(id);
        if (it != short_term_cache_.end()) {
            update_fn(it->second.memory);
            it->second.memory.updated_at = MemoryTrace::now();
            it->second.stats.recordAccess();
            writeToStm(it->second);
            return Result<bool>(true);
        }
    }
    
    // 3. 短期记忆磁盘或长期记忆
    MemoryEntry entry;
    bool found = false;
    
    {
        auto result = readFromStm(id);
        if (result.isOk() && result.value().has_value()) {
            entry = result.value().value();
            found = true;
        }
    }
    
    if (!found) {
        auto result = readFromLtm(id);
        if (result.isOk() && result.value().has_value()) {
            entry.memory = result.value().value();
            found = true;
        }
    }
    
    if (!found) {
        return Result<bool>(ErrorCode::STORAGE_NOT_FOUND,
            std::format("Memory not found: {}", id));
    }
    
    update_fn(entry.memory);
    entry.memory.updated_at = MemoryTrace::now();
    entry.stats.recordAccess();
    
    // 写回存储
    auto result = writeToLtm(entry);
    return result;
}

Result<bool> HierarchicalMemoryStore::remove(const MemoryId& id) {
    bool removed = false;
    
    // 1. 从工作记忆删除
    {
        std::unique_lock<std::shared_mutex> lock(wm_mutex_);
        auto it = working_memory_.find(id);
        if (it != working_memory_.end()) {
            // 从索引中移除
            if (it->second.memory.location.has_value()) {
                removeFromSpatialIndex(id);
            }
            if (it->second.memory.event_time.has_value()) {
                removeFromTemporalIndex(id);
            }
            if (it->second.memory.embedding.has_value()) {
                removeFromHnswIndex(id);
            }
            
            working_memory_.erase(it);
            removed = true;
        }
    }
    
    // 2. 从短期记忆删除
    {
        std::unique_lock<std::shared_mutex> lock(stm_mutex_);
        short_term_cache_.erase(id);
    }
    
    auto stm_result = deleteFromStm(id);
    if (stm_result.isOk()) {
        removed = true;
    }
    
    // 3. 从长期记忆删除
    auto ltm_result = deleteFromLtm(id);
    if (ltm_result.isOk()) {
        removed = true;
    }
    
    // 4. 从索引删除
    removeFromHnswIndex(id);
    removeFromSpatialIndex(id);
    removeFromTemporalIndex(id);
    
    return Result<bool>(removed);
}

Result<size_t> HierarchicalMemoryStore::removeBatch(const std::vector<MemoryId>& ids) {
    size_t count = 0;
    for (const auto& id : ids) {
        auto result = remove(id);
        if (result.isOk() && result.value()) {
            count++;
        }
    }
    return Result<size_t>(count);
}
