#include "processors/wake_word_detector.h"
#include <algorithm>
#include <iostream>

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
    
    std::cerr << "[LOG] Loaded " << wakeWord_ << " model" << std::endl;
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
                          std::mutex& outputMutex) {
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
            processPrediction(probability, outputMutex);
            
            // Remove one embedding worth of features
            todoFeatures_.erase(todoFeatures_.begin(),
                               todoFeatures_.begin() + EMBEDDING_FEATURES);
            
            numBufferedFeatures = todoFeatures_.size() / EMBEDDING_FEATURES;
        }
    }
}

void WakeWordDetector::processPrediction(float probability, std::mutex& outputMutex) {
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
                std::cout << wakeWord_ << std::endl;
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