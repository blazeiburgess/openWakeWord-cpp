#ifndef RING_BUFFER_H
#define RING_BUFFER_H

#include <vector>
#include <cstring>
#include <algorithm>
#include <stdexcept>

namespace openwakeword {

template<typename T>
class RingBuffer {
public:
    explicit RingBuffer(size_t capacity) 
        : buffer_(capacity), 
          capacity_(capacity),
          write_pos_(0),
          read_pos_(0),
          size_(0) {}

    void push(const T* data, size_t count) {
        if (count > available()) {
            throw std::overflow_error("Ring buffer overflow");
        }

        size_t first_chunk = std::min(count, capacity_ - write_pos_);
        std::copy(data, data + first_chunk, buffer_.data() + write_pos_);
        
        if (count > first_chunk) {
            std::copy(data + first_chunk, data + count, buffer_.data());
        }
        
        write_pos_ = (write_pos_ + count) % capacity_;
        size_ += count;
    }

    void push(const std::vector<T>& data) {
        push(data.data(), data.size());
    }

    bool pop(T* output, size_t count) {
        if (count > size_) {
            return false;
        }

        size_t first_chunk = std::min(count, capacity_ - read_pos_);
        std::copy(buffer_.data() + read_pos_, buffer_.data() + read_pos_ + first_chunk, output);
        
        if (count > first_chunk) {
            std::copy(buffer_.data(), buffer_.data() + (count - first_chunk), output + first_chunk);
        }
        
        read_pos_ = (read_pos_ + count) % capacity_;
        size_ -= count;
        return true;
    }

    bool pop(std::vector<T>& output, size_t count) {
        output.resize(count);
        return pop(output.data(), count);
    }

    // Peek without removing
    bool peek(T* output, size_t count, size_t offset = 0) const {
        if (offset + count > size_) {
            return false;
        }

        size_t peek_pos = (read_pos_ + offset) % capacity_;
        size_t first_chunk = std::min(count, capacity_ - peek_pos);
        std::copy(buffer_.data() + peek_pos, buffer_.data() + peek_pos + first_chunk, output);
        
        if (count > first_chunk) {
            std::copy(buffer_.data(), buffer_.data() + (count - first_chunk), output + first_chunk);
        }
        
        return true;
    }

    // Skip elements without copying
    void skip(size_t count) {
        if (count > size_) {
            throw std::underflow_error("Ring buffer underflow");
        }
        read_pos_ = (read_pos_ + count) % capacity_;
        size_ -= count;
    }

    void clear() {
        read_pos_ = 0;
        write_pos_ = 0;
        size_ = 0;
    }

    size_t size() const { return size_; }
    size_t capacity() const { return capacity_; }
    size_t available() const { return capacity_ - size_; }
    bool empty() const { return size_ == 0; }
    bool full() const { return size_ == capacity_; }

private:
    std::vector<T> buffer_;
    size_t capacity_;
    size_t write_pos_;
    size_t read_pos_;
    size_t size_;
};

} // namespace openwakeword

#endif // RING_BUFFER_H