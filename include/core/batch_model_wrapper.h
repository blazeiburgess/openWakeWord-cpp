#ifndef BATCH_MODEL_WRAPPER_H
#define BATCH_MODEL_WRAPPER_H

#include "core/model_wrapper.h"
#include <queue>
#include <chrono>
#include <mutex>
#include <functional>

namespace openwakeword {

// Enhanced model wrapper with batch inference support
template<typename InputType, typename OutputType>
class BatchModelWrapper : public ModelWrapper {
public:
    BatchModelWrapper(const std::string& modelName, ModelType type, size_t maxBatchSize = 8)
        : ModelWrapper(modelName, type), maxBatchSize_(maxBatchSize) {}
    
    // Single inference (backwards compatible)
    virtual OutputType inference(const InputType& input) = 0;
    
    // Batch inference
    virtual std::vector<OutputType> batchInference(const std::vector<InputType>& inputs) {
        // Default implementation: process one by one
        std::vector<OutputType> outputs;
        outputs.reserve(inputs.size());
        for (const auto& input : inputs) {
            outputs.push_back(inference(input));
        }
        return outputs;
    }
    
protected:
    size_t maxBatchSize_;
};

// Batch-enabled mel spectrogram model
class BatchMelSpectrogramModel : public BatchModelWrapper<AudioBuffer, MelBuffer> {
public:
    BatchMelSpectrogramModel();
    
    MelBuffer inference(const AudioBuffer& samples) override;
    std::vector<MelBuffer> batchInference(const std::vector<AudioBuffer>& sampleBatches) override;
    
    // Legacy interface for compatibility
    MelBuffer computeMelSpectrogram(const AudioBuffer& samples) { 
        return inference(samples); 
    }
    
private:
    size_t frameSize_ = 4 * CHUNK_SAMPLES;
};

// Batch-enabled embedding model
class BatchEmbeddingModel : public BatchModelWrapper<MelBuffer, FeatureBuffer> {
public:
    BatchEmbeddingModel();
    
    FeatureBuffer inference(const MelBuffer& mels) override;
    std::vector<FeatureBuffer> batchInference(const std::vector<MelBuffer>& melBatches) override;
    
    // Legacy interface for compatibility
    FeatureBuffer extractEmbeddings(const MelBuffer& mels) { 
        return inference(mels); 
    }
};

// Batch-enabled wake word model with adaptive batching
class BatchWakeWordModel : public BatchModelWrapper<FeatureBuffer, float> {
public:
    BatchWakeWordModel(const std::string& wakeWord);
    
    float inference(const FeatureBuffer& features) override;
    std::vector<float> batchInference(const std::vector<FeatureBuffer>& featureBatches) override;
    
    // Legacy interface for compatibility
    float predict(const FeatureBuffer& features) { 
        return inference(features); 
    }
    
    const std::string& getWakeWord() const { return wakeWord_; }
    
private:
    std::string wakeWord_;
};

// Adaptive batch processor for real-time inference
template<typename InputType, typename OutputType>
class AdaptiveBatchProcessor {
public:
    using ModelPtr = std::shared_ptr<BatchModelWrapper<InputType, OutputType>>;
    using ResultCallback = std::function<void(size_t index, OutputType result)>;
    
    AdaptiveBatchProcessor(ModelPtr model, size_t maxBatchSize = 8, 
                          std::chrono::milliseconds maxLatency = std::chrono::milliseconds(10))
        : model_(model), maxBatchSize_(maxBatchSize), maxLatency_(maxLatency) {}
    
    void process(InputType input, size_t index, ResultCallback callback) {
        {
            std::lock_guard<std::mutex> lock(queueMutex_);
            queue_.emplace(std::move(input), index, callback);
        }
        
        tryProcessBatch();
    }
    
private:
    struct Request {
        InputType input;
        size_t index;
        ResultCallback callback;
        std::chrono::steady_clock::time_point timestamp;
        
        Request(InputType&& in, size_t idx, ResultCallback cb)
            : input(std::move(in)), index(idx), callback(cb),
              timestamp(std::chrono::steady_clock::now()) {}
    };
    
    void tryProcessBatch() {
        std::vector<Request> batch;
        
        {
            std::lock_guard<std::mutex> lock(queueMutex_);
            
            auto now = std::chrono::steady_clock::now();
            
            // Collect batch based on size and latency constraints
            while (!queue_.empty() && batch.size() < maxBatchSize_) {
                auto& front = queue_.front();
                
                // Check if we should process due to latency
                if (batch.empty() || 
                    (now - front.timestamp) >= maxLatency_ ||
                    batch.size() + 1 == maxBatchSize_) {
                    
                    batch.push_back(std::move(queue_.front()));
                    queue_.pop();
                } else {
                    break;
                }
            }
        }
        
        if (!batch.empty()) {
            processBatch(std::move(batch));
        }
    }
    
    void processBatch(std::vector<Request> batch) {
        // Extract inputs
        std::vector<InputType> inputs;
        inputs.reserve(batch.size());
        for (auto& req : batch) {
            inputs.push_back(std::move(req.input));
        }
        
        // Run batch inference
        auto outputs = model_->batchInference(inputs);
        
        // Dispatch results
        for (size_t i = 0; i < batch.size(); ++i) {
            batch[i].callback(batch[i].index, std::move(outputs[i]));
        }
    }
    
    ModelPtr model_;
    size_t maxBatchSize_;
    std::chrono::milliseconds maxLatency_;
    
    std::mutex queueMutex_;
    std::queue<Request> queue_;
};

} // namespace openwakeword

#endif // BATCH_MODEL_WRAPPER_H