#include "processors/wake_word_detector.h"
#include <algorithm>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <sstream>

namespace openwakeword {

WakeWordDetector::WakeWordDetector(const std::string& wakeWord, 
                                   const WakeWordConfig& config,
                                   Ort::Env& env, 
                                   const Ort::SessionOptions& options)
    : AudioProcessor(wakeWord), wakeWord_(wakeWord), config_(config), 
      env_(env), options_(options) {
}

bool WakeWordDetector::initialize() {
    if (!std::filesystem::exists(config_.modelPath)) {
        std::cerr << "[ERROR] Wake word model not found: " << config_.modelPath << std::endl;
        return false;
    }
    
    model_ = std::make_unique<WakeWordModel>(wakeWord_);
    if (!model_->loadModel(config_.modelPath, env_, options_)) {
        std::cerr << "[ERROR] Failed to load wake word model: " << wakeWord_ << std::endl;
        return false;
    }
    
    // Log message handled by pipeline based on output mode
    initialized_ = true;
    return true;
}

bool WakeWordDetector::process() {
    // This method is not used in threaded mode
    return true;
}

void WakeWordDetector::reset() {
    todoFeatures_.clear();
    activationCount_ = 0;
}

void WakeWordDetector::run(std::shared_ptr<ThreadSafeBuffer<AudioFloat>> input,
                          std::mutex& outputMutex,
                          OutputMode outputMode,
                          bool showTimestamp) {
    if (!initialized_) {
        std::cerr << "[ERROR] WakeWordDetector not initialized: " << wakeWord_ << std::endl;
        return;
    }
    
    while (true) {
        // Get features from input buffer
        auto features = input->pull();
        if (input->isExhausted() && features.empty()) {
            break;
        }
        
        // Accumulate features
        todoFeatures_.insert(todoFeatures_.end(), features.begin(), features.end());
        
        // Process when we have enough features
        size_t numBufferedFeatures = todoFeatures_.size() / EMBEDDING_FEATURES;
        while (numBufferedFeatures >= WAKEWORD_FEATURES) {
            // Extract features for one prediction
            FeatureBuffer windowFeatures(todoFeatures_.begin(),
                                       todoFeatures_.begin() + (WAKEWORD_FEATURES * EMBEDDING_FEATURES));
            
            // Run wake word detection
            float probability = model_->predict(windowFeatures);
            
            // Process the prediction
            processPrediction(probability, outputMutex, outputMode, showTimestamp);
            
            // Remove one embedding worth of features
            todoFeatures_.erase(todoFeatures_.begin(),
                               todoFeatures_.begin() + EMBEDDING_FEATURES);
            
            numBufferedFeatures = todoFeatures_.size() / EMBEDDING_FEATURES;
        }
    }
}

void WakeWordDetector::processPrediction(float probability, std::mutex& outputMutex,
                                       OutputMode outputMode, bool showTimestamp) {
    if (config_.debug) {
        std::unique_lock<std::mutex> lock(outputMutex);
        std::cerr << wakeWord_ << " " << probability << std::endl;
    }
    
    if (probability > config_.threshold) {
        // Activation detected
        activationCount_++;
        
        if (activationCount_ >= config_.triggerLevel) {
            // Trigger level reached - output detection
            {
                std::unique_lock<std::mutex> lock(outputMutex);
                
                if (outputMode == OutputMode::JSON) {
                    // JSON output
                    std::cout << "{";
                    std::cout << "\"wake_word\":\"" << wakeWord_ << "\"";
                    std::cout << ",\"score\":" << probability;
                    if (showTimestamp) {
                        auto now = std::chrono::system_clock::now();
                        auto time_t = std::chrono::system_clock::to_time_t(now);
                        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                            now.time_since_epoch()) % 1000;
                        
                        std::stringstream ss;
                        ss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
                        ss << "." << std::setfill('0') << std::setw(3) << ms.count();
                        
                        std::cout << ",\"timestamp\":\"" << ss.str() << "\"";
                    }
                    std::cout << "}" << std::endl;
                } else {
                    // Normal output
                    if (showTimestamp) {
                        auto now = std::chrono::system_clock::now();
                        auto time_t = std::chrono::system_clock::to_time_t(now);
                        std::cout << "[" << std::put_time(std::localtime(&time_t), "%H:%M:%S") << "] ";
                    }
                    std::cout << wakeWord_ << std::endl;
                }
            }
            
            // Enter refractory period
            activationCount_ = -config_.refractorySteps;
        }
    } else {
        // No activation - decay activation count
        if (activationCount_ > 0) {
            activationCount_ = std::max(0, activationCount_ - 1);
        } else {
            activationCount_ = std::min(0, activationCount_ + 1);
        }
    }
}

} // namespace openwakeword