#ifndef LOCK_FREE_QUEUE_H
#define LOCK_FREE_QUEUE_H

#include <atomic>
#include <memory>
#include <vector>

namespace openwakeword {

template<typename T>
class LockFreeQueue {
private:
    struct Node {
        std::atomic<T*> data;
        std::atomic<Node*> next;
        
        Node() : data(nullptr), next(nullptr) {}
    };
    
    std::atomic<Node*> head_;
    std::atomic<Node*> tail_;
    
public:
    LockFreeQueue() {
        Node* dummy = new Node;
        head_.store(dummy);
        tail_.store(dummy);
    }
    
    ~LockFreeQueue() {
        // Clean up remaining nodes
        while (Node* oldHead = head_.load()) {
            head_.store(oldHead->next);
            T* data = oldHead->data.load();
            delete data;
            delete oldHead;
        }
    }
    
    void push(T item) {
        Node* newNode = new Node;
        T* data = new T(std::move(item));
        newNode->data.store(data);
        
        Node* prevTail = tail_.exchange(newNode);
        prevTail->next.store(newNode);
    }
    
    bool try_pop(T& result) {
        Node* head = head_.load();
        Node* next = head->next.load();
        
        if (next == nullptr) {
            return false;
        }
        
        T* data = next->data.load();
        if (data == nullptr) {
            return false;
        }
        
        result = std::move(*data);
        head_.store(next);
        
        delete data;
        delete head;
        return true;
    }
    
    bool empty() const {
        Node* head = head_.load();
        Node* next = head->next.load();
        return next == nullptr;
    }
};

// Optimized lock-free queue for bulk operations
template<typename T>
class BulkLockFreeQueue {
private:
    static constexpr size_t CACHE_LINE_SIZE = 64;
    static constexpr size_t BULK_SIZE = 128;
    
    struct alignas(CACHE_LINE_SIZE) ProducerData {
        std::vector<T> buffer;
        size_t write_pos = 0;
        
        ProducerData() {
            buffer.reserve(BULK_SIZE);
        }
    };
    
    struct alignas(CACHE_LINE_SIZE) ConsumerData {
        std::vector<T> buffer;
        size_t read_pos = 0;
    };
    
    alignas(CACHE_LINE_SIZE) std::atomic<ProducerData*> producer_data_;
    alignas(CACHE_LINE_SIZE) std::atomic<ConsumerData*> consumer_data_;
    LockFreeQueue<std::vector<T>> queue_;
    
public:
    BulkLockFreeQueue() 
        : producer_data_(new ProducerData()),
          consumer_data_(new ConsumerData()) {}
    
    ~BulkLockFreeQueue() {
        delete producer_data_.load();
        delete consumer_data_.load();
    }
    
    void push(T item) {
        ProducerData* pd = producer_data_.load(std::memory_order_relaxed);
        pd->buffer.push_back(std::move(item));
        
        if (pd->buffer.size() >= BULK_SIZE) {
            flush();
        }
    }
    
    void push(const std::vector<T>& items) {
        if (items.size() >= BULK_SIZE) {
            // Large batch, push directly
            queue_.push(items);
        } else {
            // Small batch, buffer it
            ProducerData* pd = producer_data_.load(std::memory_order_relaxed);
            pd->buffer.insert(pd->buffer.end(), items.begin(), items.end());
            
            if (pd->buffer.size() >= BULK_SIZE) {
                flush();
            }
        }
    }
    
    void flush() {
        ProducerData* pd = producer_data_.load(std::memory_order_relaxed);
        if (!pd->buffer.empty()) {
            queue_.push(std::move(pd->buffer));
            pd->buffer.clear();
            pd->buffer.reserve(BULK_SIZE);
        }
    }
    
    bool try_pop(T& result) {
        ConsumerData* cd = consumer_data_.load(std::memory_order_relaxed);
        
        // Check local buffer first
        if (cd->read_pos < cd->buffer.size()) {
            result = std::move(cd->buffer[cd->read_pos++]);
            return true;
        }
        
        // Refill from queue
        cd->buffer.clear();
        cd->read_pos = 0;
        
        if (queue_.try_pop(cd->buffer)) {
            if (!cd->buffer.empty()) {
                result = std::move(cd->buffer[cd->read_pos++]);
                return true;
            }
        }
        
        return false;
    }
    
    std::vector<T> try_pop_bulk(size_t max_items) {
        ConsumerData* cd = consumer_data_.load(std::memory_order_relaxed);
        std::vector<T> results;
        results.reserve(max_items);
        
        // Get from local buffer first
        while (cd->read_pos < cd->buffer.size() && results.size() < max_items) {
            results.push_back(std::move(cd->buffer[cd->read_pos++]));
        }
        
        // Try to get more from queue
        while (results.size() < max_items) {
            cd->buffer.clear();
            cd->read_pos = 0;
            
            if (!queue_.try_pop(cd->buffer)) {
                break;
            }
            
            size_t to_take = std::min(cd->buffer.size(), max_items - results.size());
            results.insert(results.end(), 
                         std::make_move_iterator(cd->buffer.begin()),
                         std::make_move_iterator(cd->buffer.begin() + to_take));
            cd->read_pos = to_take;
        }
        
        return results;
    }
    
    bool empty() {
        ConsumerData* cd = consumer_data_.load(std::memory_order_relaxed);
        return cd->read_pos >= cd->buffer.size() && queue_.empty();
    }
};

} // namespace openwakeword

#endif // LOCK_FREE_QUEUE_H