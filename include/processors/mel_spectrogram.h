#ifndef OPENWAKEWORD_MEL_SPECTROGRAM_H
#define OPENWAKEWORD_MEL_SPECTROGRAM_H

#include <condition_variable>
#include <filesystem>
#include <memory>
#include <mutex>
#include <vector>
#include "core/audio_processor.h"
#include "core/model_wrapper.h"
#include "core/types.h"

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

// Mel spectrogram processor
class MelSpectrogramProcessor : public TransformProcessor<AudioFloat, AudioFloat> {
public:
    MelSpectrogramProcessor(Ort::Env& env, const Ort::SessionOptions& options);
    
    // Set model path
    void setModelPath(const std::filesystem::path& path) { modelPath_ = path; }
    
    // Set frame size
    void setFrameSize(size_t frameSize) { frameSize_ = frameSize; }
    
    // AudioProcessor interface
    bool initialize() override;
    bool process() override;
    void reset() override;
    
    // Thread entry point
    void run(std::shared_ptr<ThreadSafeBuffer<AudioFloat>> input,
             std::shared_ptr<ThreadSafeBuffer<AudioFloat>> output);
    
private:
    Ort::Env& env_;
    Ort::SessionOptions options_;
    std::filesystem::path modelPath_;
    size_t frameSize_ = 4 * CHUNK_SAMPLES;
    
    std::unique_ptr<MelSpectrogramModel> model_;
    std::vector<AudioFloat> todoSamples_;
};

} // namespace openwakeword

#endif // OPENWAKEWORD_MEL_SPECTROGRAM_H