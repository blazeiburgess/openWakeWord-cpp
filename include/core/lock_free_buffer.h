#ifndef LOCK_FREE_BUFFER_H
#define LOCK_FREE_BUFFER_H

#include <atomic>
#include <vector>
#include <condition_variable>
#include <mutex>
#include "core/lock_free_queue.h"

namespace openwakeword {

// Lock-free buffer with same interface as ThreadSafeBuffer
template<typename T>
class LockFreeBuffer {
private:
    BulkLockFreeQueue<T> queue_;
    std::atomic<bool> exhausted_{false};
    
    // Fallback notification mechanism for blocking operations
    mutable std::mutex cv_mutex_;
    mutable std::condition_variable cv_;
    
public:
    LockFreeBuffer() = default;
    
    void push(const std::vector<T>& data) {
        if (!exhausted_.load(std::memory_order_acquire)) {
            queue_.push(data);
            queue_.flush();
            cv_.notify_one();
        }
    }
    
    void push(std::vector<T>&& data) {
        if (!exhausted_.load(std::memory_order_acquire)) {
            queue_.push(std::move(data));
            queue_.flush();
            cv_.notify_one();
        }
    }
    
    std::vector<T> pull() {
        // Try non-blocking first
        std::vector<T> result = queue_.try_pop_bulk(1024);
        if (!result.empty()) {
            return result;
        }
        
        // If exhausted and empty, return empty
        if (exhausted_.load(std::memory_order_acquire) && queue_.empty()) {
            return result;
        }
        
        // Fall back to blocking wait
        std::unique_lock<std::mutex> lock(cv_mutex_);
        cv_.wait(lock, [this] {
            return !queue_.empty() || exhausted_.load(std::memory_order_acquire);
        });
        
        return queue_.try_pop_bulk(1024);
    }
    
    bool isExhausted() const {
        return exhausted_.load(std::memory_order_acquire) && queue_.empty();
    }
    
    void setExhausted(bool value) {
        exhausted_.store(value, std::memory_order_release);
        if (value) {
            queue_.flush();
            cv_.notify_all();
        }
    }
};

// Enable lock-free buffers with a compile-time flag
#ifdef USE_LOCK_FREE_BUFFERS
    template<typename T>
    using ThreadSafeBufferOptimized = LockFreeBuffer<T>;
#else
    template<typename T>
    using ThreadSafeBufferOptimized = ThreadSafeBuffer<T>;
#endif

} // namespace openwakeword

#endif // LOCK_FREE_BUFFER_H