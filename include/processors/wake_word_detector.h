#ifndef OPENWAKEWORD_WAKE_WORD_DETECTOR_H
#define OPENWAKEWORD_WAKE_WORD_DETECTOR_H

#include <filesystem>
#include <memory>
#include <mutex>
#include <vector>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <sstream>
#include "core/audio_processor.h"
#include "core/model_wrapper.h"
#include "core/types.h"
#include "core/thread_safe_buffer.h"
#include "utils/config.h"
#include "utils/ring_buffer.h"

namespace openwakeword {

// Configuration for wake word detection
struct WakeWordConfig {
    std::filesystem::path modelPath;
    float threshold = 0.5f;
    int triggerLevel = 4;    // Number of consecutive activations needed
    int refractorySteps = 20; // Steps to wait after activation
    bool debug = false;
};

// Wake word detector processor
class WakeWordDetector : public AudioProcessor {
public:
    WakeWordDetector(const std::string& wakeWord, const WakeWordConfig& config,
                     Ort::Env& env, const Ort::SessionOptions& options);
    
    // AudioProcessor interface
    bool initialize() override;
    bool process() override;
    void reset() override;
    
    // Thread entry point
    template<typename BufferType>
    void run(std::shared_ptr<BufferType> input,
             std::mutex& outputMutex,
             OutputMode outputMode,
             bool showTimestamp);
    
    // Get configuration
    const WakeWordConfig& getConfig() const { return config_; }
    
private:
    std::string wakeWord_;
    WakeWordConfig config_;
    Ort::Env& env_;
    const Ort::SessionOptions& options_;
    
    std::unique_ptr<WakeWordModel> model_;
    RingBuffer<AudioFloat> featureBuffer_;
    
    // Activation tracking
    int activationCount_ = 0;
    
    // Process a single prediction
    void processPrediction(float probability, std::mutex& outputMutex,
                          OutputMode outputMode, bool showTimestamp);
};

// Template implementation
template<typename BufferType>
void WakeWordDetector::run(std::shared_ptr<BufferType> input,
                          std::mutex& outputMutex,
                          OutputMode outputMode,
                          bool showTimestamp) {
    if (!initialized_) {
        if (outputMode != OutputMode::QUIET) {
            std::cerr << "[ERROR] WakeWordDetector not initialized: " << wakeWord_ << std::endl;
        }
        return;
    }
    
    while (true) {
        // Get features from input buffer
        auto features = input->pull();
        if (input->isExhausted() && features.empty()) {
            break;
        }
        
        // Push features to ring buffer
        if (!features.empty()) {
            featureBuffer_.push(features);
        }
        
        // Process when we have enough features
        size_t numBufferedFeatures = featureBuffer_.size() / EMBEDDING_FEATURES;
        while (numBufferedFeatures >= WAKEWORD_FEATURES) {
            // Pre-allocated buffer for features
            thread_local FeatureBuffer windowFeatures(WAKEWORD_FEATURES * EMBEDDING_FEATURES);
            
            // Peek features for one prediction (sliding window)
            featureBuffer_.peek(windowFeatures.data(), WAKEWORD_FEATURES * EMBEDDING_FEATURES);
            
            // Run wake word detection
            float probability = model_->predict(windowFeatures);
            
            // Process the prediction
            processPrediction(probability, outputMutex, outputMode, showTimestamp);
            
            // Slide window by one embedding
            featureBuffer_.skip(EMBEDDING_FEATURES);
            
            numBufferedFeatures = featureBuffer_.size() / EMBEDDING_FEATURES;
        }
    }
}

} // namespace openwakeword

#endif // OPENWAKEWORD_WAKE_WORD_DETECTOR_H