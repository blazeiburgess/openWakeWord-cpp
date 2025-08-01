#ifndef OPENWAKEWORD_THREAD_SAFE_BUFFER_H
#define OPENWAKEWORD_THREAD_SAFE_BUFFER_H

#include <condition_variable>
#include <mutex>
#include <vector>

namespace openwakeword {

// Thread-safe buffer for inter-processor communication
template<typename T>
class ThreadSafeBuffer {
public:
    void push(const std::vector<T>& data) {
        std::unique_lock<std::mutex> lock(mutex_);
        buffer_.insert(buffer_.end(), data.begin(), data.end());
        ready_ = true;
        cv_.notify_one();
    }
    
    void push(const T* data, size_t count) {
        std::unique_lock<std::mutex> lock(mutex_);
        buffer_.insert(buffer_.end(), data, data + count);
        ready_ = true;
        cv_.notify_one();
    }
    
    std::vector<T> pull(size_t maxCount = 0) {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this] { return ready_ || exhausted_; });
        
        if (exhausted_ && buffer_.empty()) {
            return {};
        }
        
        std::vector<T> result;
        if (maxCount == 0 || maxCount >= buffer_.size()) {
            result = std::move(buffer_);
            buffer_.clear();
        } else {
            result.assign(buffer_.begin(), buffer_.begin() + maxCount);
            buffer_.erase(buffer_.begin(), buffer_.begin() + maxCount);
        }
        
        if (!exhausted_) {
            ready_ = false;
        }
        
        return result;
    }
    
    void setExhausted(bool exhausted) {
        std::unique_lock<std::mutex> lock(mutex_);
        exhausted_ = exhausted;
        ready_ = true;
        cv_.notify_one();
    }
    
    bool isExhausted() const {
        std::unique_lock<std::mutex> lock(mutex_);
        return exhausted_ && buffer_.empty();
    }
    
    size_t size() const {
        std::unique_lock<std::mutex> lock(mutex_);
        return buffer_.size();
    }
    
private:
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::vector<T> buffer_;
    bool ready_ = false;
    bool exhausted_ = false;
};

} // namespace openwakeword

#endif // OPENWAKEWORD_THREAD_SAFE_BUFFER_H