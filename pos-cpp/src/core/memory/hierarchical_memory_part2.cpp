/**
 * @file hierarchical_memory.cpp (Part 2)
 * @brief 四层分层记忆存储系统实现 - Layer 0 & 1
 * 感知记忆和工作记忆操作
 */

#include "hierarchical_memory.h"
#include <chrono>
#include <algorithm>

namespace personal_ontology {
namespace memory {

// =============================================================================
// Layer 0: 感知记忆操作
// =============================================================================

Result<MemoryId> QuadLayerMemoryStore::sensoryInput(
    SensoryType type,
    const std::vector<uint8_t>& raw_data,
    float attention_hint) {
    
    SensoryBufferEntry entry(type);
    entry.raw_data = raw_data;
    entry.attention_weight = attention_hint;
    
    {
        std::unique_lock<std::shared_mutex> lock(sensory_mutex_);
        
        // 如果缓冲区已满，移除最旧的条目
        if (sensory_buffer_.size() >= config_.sensory.buffer_size) {
            sensory_buffer_.pop_front();
        }
        
        sensory_buffer_.push_back(std::move(entry));
    }
    
    // 自动注意机制：高权重条目直接进入工作记忆
    if (attention_hint >= config_.sensory.attention_threshold) {
        // 这里会调用 attend() 将感知输入提升到工作记忆
        // 暂时返回ID，实际内容需要经过处理
    }
    
    return Result<MemoryId>(entry.id);
}

Result<MemoryId> QuadLayerMemoryStore::sensoryInputText(
    const std::string& text,
    float attention_hint) {
    
    std::vector<uint8_t> data(text.begin(), text.end());
    return sensoryInput(SensoryType::TEXT, data, attention_hint);
}

std::vector<SensoryBufferEntry> QuadLayerMemoryStore::getAttentionFocus() {
    std::vector<SensoryBufferEntry> result;
    
    std::shared_lock<std::shared_mutex> lock(sensory_mutex_);
    std::lock_guard<std::mutex> att_lock(attention_mutex_);
    
    for (const auto& entry : sensory_buffer_) {
        if (attention_focus_.count(entry.id) > 0 || 
            entry.attention_weight >= config_.sensory.attention_threshold) {
            result.push_back(entry);
        }
    }
    
    return result;
}

void QuadLayerMemoryStore::triggerAttention(const std::vector<MemoryId>& ids) {
    std::lock_guard<std::mutex> lock(attention_mutex_);
    for (const auto& id : ids) {
        attention_focus_.insert(id);
    }
}

void QuadLayerMemoryStore::decaySensoryMemory() {
    std::unique_lock<std::shared_mutex> lock(sensory_mutex_);
    std::lock_guard<std::mutex> att_lock(attention_mutex_);
    
    auto now = MemoryTrace::now();
    
    // 移除过期的条目
    sensory_buffer_.erase(
        std::remove_if(sensory_buffer_.begin(), sensory_buffer_.end(),
            [&](const SensoryBufferEntry& entry) {
                bool expired = entry.isExpired(now, config_.sensory.decay_ms);
                if (expired) {
                    attention_focus_.erase(entry.id);
                }
                return expired;
            }),
        sensory_buffer_.end()
    );
}

// =============================================================================
// Layer 1: 工作记忆操作
// =============================================================================

Result<bool> QuadLayerMemoryStore::loadToWorkingMemory(
    const MemoryTrace& memory,
    bool set_focus) {
    
    std::unique_lock<std::shared_mutex> lock(working_mutex_);
    
    // 检查容量
    if (working_memory_.size() >= config_.working.capacity) {
        // 找到激活水平最低的条目并移除
        auto it = std::min_element(working_memory_.begin(), working_memory_.end(),
            [](const auto& a, const auto& b) {
                return a.second.activation_level < b.second.activation_level;
            });
        
        if (it != working_memory_.end()) {
            // 尝试巩固到长期记忆
            if (it->second.activation_level > 0.3f) {
                // 高激活的应该巩固
                lock.unlock();  // 先解锁避免死锁
                consolidate(it->first);
                lock.lock();
            }
            working_memory_.erase(it);
        }
    }
    
    // 添加新条目
    MemoryTrace mem_copy = memory;
    if (mem_copy.id.empty()) {
        mem_copy.id = generateUUID();
    }
    mem_copy.created_at = MemoryTrace::now();
    mem_copy.updated_at = mem_copy.created_at;
    
    WorkingMemoryEntry entry(mem_copy);
    entry.is_focused = set_focus;
    
    if (set_focus) {
        // 清除之前的焦点
        for (auto& [id, wm_entry] : working_memory_) {
            wm_entry.is_focused = false;
        }
        focused_id_ = mem_copy.id;
    }
    
    working_memory_[mem_copy.id] = std::move(entry);
    
    return Result<bool>(true);
}

std::optional<MemoryTrace> QuadLayerMemoryStore::getFocusedMemory() {
    std::shared_lock<std::shared_mutex> lock(working_mutex_);
    
    if (!focused_id_.has_value()) {
        return std::nullopt;
    }
    
    auto it = working_memory_.find(focused_id_.value());
    if (it != working_memory_.end()) {
        it->second.stats.recordAccess();
        return it->second.memory;
    }
    
    return std::nullopt;
}

void QuadLayerMemoryStore::setFocus(const MemoryId& id) {
    std::unique_lock<std::shared_mutex> lock(working_mutex_);
    
    // 清除之前的焦点
    for (auto& [mem_id, entry] : working_memory_) {
        entry.is_focused = false;
    }
    
    auto it = working_memory_.find(id);
    if (it != working_memory_.end()) {
        it->second.is_focused = true;
        focused_id_ = id;
        it->second.stats.recordAccess();
    }
}

Result<bool> QuadLayerMemoryStore::rehearse(const MemoryId& id) {
    std::unique_lock<std::shared_mutex> lock(working_mutex_);
    
    auto it = working_memory_.find(id);
    if (it == working_memory_.end()) {
        return Result<bool>(ErrorCode::STORAGE_NOT_FOUND,
            std::format("Memory not in working memory: {}", id));
    }
    
    // 增加复述计数和激活水平
    it->second.rehearsal_count++;
    it->second.activation_level = std::min(1.0f, 
        it->second.activation_level + 0.1f);
    it->second.timestamp = MemoryTrace::now();
    
    return Result<bool>(true);
}

std::vector<WorkingMemoryEntry> QuadLayerMemoryStore::getWorkingMemoryContents() {
    std::shared_lock<std::shared_mutex> lock(working_mutex_);
    
    std::vector<WorkingMemoryEntry> result;
    result.reserve(working_memory_.size());
    
    for (const auto& [id, entry] : working_memory_) {
        result.push_back(entry);
    }
    
    // 按激活水平排序
    std::sort(result.begin(), result.end(),
        [](const WorkingMemoryEntry& a, const WorkingMemoryEntry& b) {
            return a.activation_level > b.activation_level;
        });
    
    return result;
}

Result<MemoryId> QuadLayerMemoryStore::createChunk(
    const std::string& label,
    const std::vector<MemoryId>& items) {
    
    std::unique_lock<std::shared_mutex> lock(working_mutex_);
    
    MemoryChunk chunk;
    chunk.label = label;
    chunk.items = items;
    
    // 计算块的重要性（成员的平均重要性）
    float total_importance = 0.0f;
    for (const auto& item_id : items) {
        auto it = working_memory_.find(item_id);
        if (it != working_memory_.end()) {
            total_importance += it->second.activation_level;
            it->second.chunk = chunk;
        }
    }
    chunk.importance = items.empty() ? 0.5f : total_importance / items.size();
    
    chunks_[chunk.id] = std::move(chunk);
    
    return Result<MemoryId>(chunk.id);
}

void QuadLayerMemoryStore::decayWorkingMemory() {
    std::unique_lock<std::shared_mutex> lock(working_mutex_);
    
    auto now = MemoryTrace::now();
    std::vector<MemoryId> to_consolidate;
    
    // 遍历并更新激活水平
    for (auto it = working_memory_.begin(); it != working_memory_.end();) {
        auto& entry = it->second;
        
        // 计算衰减后的激活水平
        float new_activation = entry.computeActivation(now, config_.working.max_age_ms);
        
        if (new_activation < 0.1f && !entry.is_focused) {
            // 应该被移除
            if (entry.rehearsal_count > 0) {
                // 曾经被复述过，考虑巩固
                to_consolidate.push_back(it->first);
            }
            
            if (focused_id_ == it->first) {
                focused_id_ = std::nullopt;
            }
            it = working_memory_.erase(it);
        } else {
            entry.activation_level = new_activation;
            ++it;
        }
    }
    
    lock.unlock();
    
    // 巩固有价值的记忆
    for (const auto& id : to_consolidate) {
        consolidate(id);
    }
}

// =============================================================================
// 层间转换: 感知 → 工作记忆
// =============================================================================

Result<MemoryId> QuadLayerMemoryStore::attend(
    const MemoryId& sensory_id,
    const std::string& interpreted_content) {
    
    // 从感知缓冲区找到条目
    SensoryBufferEntry* sensory_entry = nullptr;
    {
        std::shared_lock<std::shared_mutex> lock(sensory_mutex_);
        for (auto& entry : sensory_buffer_) {
            if (entry.id == sensory_id) {
                sensory_entry = &entry;
                break;
            }
        }
    }
    
    if (!sensory_entry) {
        return Result<MemoryId>(ErrorCode::STORAGE_NOT_FOUND,
            std::format("Sensory entry not found: {}", sensory_id));
    }
    
    // 创建记忆痕迹
    MemoryTrace memory;
    memory.id = generateUUID();  // 工作记忆中使用新ID
    memory.content = interpreted_content;
    memory.type = "attended";
    memory.source = sensory_entry->type == SensoryType::TEXT ? "text_input" : "sensory";
    memory.timestamp = MemoryTrace::now();
    
    if (sensory_entry->embedding.has_value()) {
        memory.embedding = sensory_entry->embedding;
    }
    if (sensory_entry->location.has_value()) {
        memory.location = sensory_entry->location;
    }
    
    // 添加到工作记忆
    auto result = loadToWorkingMemory(memory, true);
    if (result.isError()) {
        return Result<MemoryId>(result.errorCode(), result.errorMessage());
    }
    
    // 添加到注意焦点
    triggerAttention({sensory_id});
    
    return Result<MemoryId>(memory.id);
}

} // namespace memory
} // namespace personal_ontology
