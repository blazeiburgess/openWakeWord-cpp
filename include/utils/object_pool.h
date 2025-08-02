#ifndef OBJECT_POOL_H
#define OBJECT_POOL_H

#include <vector>
#include <memory>
#include <functional>
#include <mutex>
#include <condition_variable>

namespace openwakeword {

template<typename T>
class ObjectPool {
public:
    using CreateFunc = std::function<std::unique_ptr<T>()>;
    using ResetFunc = std::function<void(T&)>;
    
    ObjectPool(size_t initialSize, CreateFunc createFunc, ResetFunc resetFunc = nullptr)
        : createFunc_(createFunc), resetFunc_(resetFunc) {
        // Pre-allocate objects
        for (size_t i = 0; i < initialSize; ++i) {
            pool_.push_back(createFunc_());
        }
    }
    
    // RAII wrapper for borrowed objects
    class BorrowedObject {
    public:
        BorrowedObject(ObjectPool* pool, std::unique_ptr<T> obj)
            : pool_(pool), obj_(std::move(obj)) {}
        
        ~BorrowedObject() {
            if (obj_ && pool_) {
                pool_->returnObject(std::move(obj_));
            }
        }
        
        // Move-only
        BorrowedObject(BorrowedObject&& other) noexcept
            : pool_(other.pool_), obj_(std::move(other.obj_)) {
            other.pool_ = nullptr;
        }
        
        BorrowedObject& operator=(BorrowedObject&& other) noexcept {
            if (this != &other) {
                if (obj_ && pool_) {
                    pool_->returnObject(std::move(obj_));
                }
                pool_ = other.pool_;
                obj_ = std::move(other.obj_);
                other.pool_ = nullptr;
            }
            return *this;
        }
        
        // Delete copy operations
        BorrowedObject(const BorrowedObject&) = delete;
        BorrowedObject& operator=(const BorrowedObject&) = delete;
        
        T* operator->() { return obj_.get(); }
        const T* operator->() const { return obj_.get(); }
        T& operator*() { return *obj_; }
        const T& operator*() const { return *obj_; }
        T* get() { return obj_.get(); }
        const T* get() const { return obj_.get(); }
        
    private:
        ObjectPool* pool_;
        std::unique_ptr<T> obj_;
    };
    
    BorrowedObject borrow() {
        std::unique_lock<std::mutex> lock(mutex_);
        
        // Wait if pool is empty
        while (pool_.empty()) {
            cv_.wait(lock);
        }
        
        auto obj = std::move(pool_.back());
        pool_.pop_back();
        
        return BorrowedObject(this, std::move(obj));
    }
    
    // Try to borrow without blocking
    std::unique_ptr<BorrowedObject> tryBorrow() {
        std::unique_lock<std::mutex> lock(mutex_);
        
        if (pool_.empty()) {
            return nullptr;
        }
        
        auto obj = std::move(pool_.back());
        pool_.pop_back();
        
        return std::make_unique<BorrowedObject>(this, std::move(obj));
    }
    
    size_t available() const {
        std::unique_lock<std::mutex> lock(mutex_);
        return pool_.size();
    }
    
private:
    void returnObject(std::unique_ptr<T> obj) {
        if (resetFunc_) {
            resetFunc_(*obj);
        }
        
        std::unique_lock<std::mutex> lock(mutex_);
        pool_.push_back(std::move(obj));
        cv_.notify_one();
    }
    
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::vector<std::unique_ptr<T>> pool_;
    CreateFunc createFunc_;
    ResetFunc resetFunc_;
};

// Specialized pool for vectors
template<typename T>
class VectorPool {
public:
    VectorPool(size_t poolSize, size_t vectorCapacity)
        : pool_(poolSize,
                []() { return std::make_unique<std::vector<T>>(); },
                [vectorCapacity](std::vector<T>& v) { 
                    v.clear(); 
                    v.reserve(vectorCapacity);
                }) {
        // Pre-reserve capacity for all vectors
        for (size_t i = 0; i < poolSize; ++i) {
            auto borrowed = pool_.borrow();
            borrowed->reserve(vectorCapacity);
        }
    }
    
    auto borrow() {
        return pool_.borrow();
    }
    
    auto tryBorrow() {
        return pool_.tryBorrow();
    }
    
    size_t available() const {
        return pool_.available();
    }
    
private:
    ObjectPool<std::vector<T>> pool_;
};

} // namespace openwakeword

#endif // OBJECT_POOL_H