#include "core/pipeline.h"
#include "utils/simd_audio.h"
#include <iostream>

namespace openwakeword {

Pipeline::Pipeline(const Config& config) 
    : config_(config),
      env_(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "openWakeWord") {
    env_.DisableTelemetryEvents();
    
    // Configure session options
    sessionOptions_.SetIntraOpNumThreads(config_.intraOpNumThreads);
    sessionOptions_.SetInterOpNumThreads(config_.interOpNumThreads);
    
    // Calculate expected ready count
    expectedReadyCount_ = 2 + config_.wakeWordConfigs.size(); // mel + embedding + wake words
    
    // Initialize audio buffer pool with pre-allocated buffers
    // Pool size: 4 buffers, each with capacity for max frame size
    audioBufferPool_ = std::make_unique<VectorPool<AudioFloat>>(4, config_.frameSize);
}

Pipeline::~Pipeline() {
    stop();
}

bool Pipeline::initialize() {
    // Create buffers
    audioBuffer_ = std::make_shared<ThreadSafeBuffer<AudioFloat>>();
    melBuffer_ = std::make_shared<ThreadSafeBuffer<AudioFloat>>();
    
    // Create feature buffers for each wake word detector
    for (size_t i = 0; i < config_.wakeWordConfigs.size(); ++i) {
        featureBuffers_.push_back(std::make_shared<ThreadSafeBuffer<AudioFloat>>());
    }
    
    // Initialize mel spectrogram processor
    melProcessor_ = std::make_unique<MelSpectrogramProcessor>(env_, sessionOptions_);
    melProcessor_->setModelPath(config_.melModelPath);
    melProcessor_->setFrameSize(config_.frameSize);
    if (!melProcessor_->initialize()) {
        return false;
    }
    if (config_.outputMode == OutputMode::VERBOSE || config_.outputMode == OutputMode::NORMAL) {
        std::cerr << "[LOG] Loaded mel spectrogram model" << std::endl;
    }
    
    // Initialize speech embedding processor
    embeddingProcessor_ = std::make_unique<SpeechEmbeddingProcessor>(
        env_, sessionOptions_, config_.wakeWordConfigs.size());
    embeddingProcessor_->setModelPath(config_.embModelPath);
    if (!embeddingProcessor_->initialize()) {
        return false;
    }
    if (config_.outputMode == OutputMode::VERBOSE || config_.outputMode == OutputMode::NORMAL) {
        std::cerr << "[LOG] Loaded speech embedding model" << std::endl;
    }
    
    // Initialize wake word detectors
    for (const auto& wwConfig : config_.wakeWordConfigs) {
        auto wakeWord = wwConfig.modelPath.stem().string();
        auto detector = std::make_unique<WakeWordDetector>(
            wakeWord, wwConfig, env_, sessionOptions_);
        if (!detector->initialize()) {
            return false;
        }
        if (config_.outputMode == OutputMode::VERBOSE || config_.outputMode == OutputMode::NORMAL) {
            std::cerr << "[LOG] Loaded wake word model: " << wakeWord << std::endl;
        }
        detectors_.push_back(std::move(detector));
    }
    
    // Log SIMD availability
    if (config_.outputMode == OutputMode::VERBOSE) {
        std::cerr << "[LOG] SIMD audio conversion: " 
                  << (SimdAudio::isSimdAvailable() ? "enabled" : "disabled") << std::endl;
    }
    
    return true;
}

void Pipeline::start() {
    if (running_) {
        return;
    }
    
    running_ = true;
    readyCount_ = 0;
    
    // Start mel spectrogram thread
    melThread_ = std::thread([this]() {
        incrementReady();
        melProcessor_->run(audioBuffer_, melBuffer_, config_.outputMode);
    });
    
    // Start embedding thread
    embeddingThread_ = std::thread([this]() {
        incrementReady();
        embeddingProcessor_->run(melBuffer_, featureBuffers_, config_.outputMode);
    });
    
    // Start wake word detector threads
    for (size_t i = 0; i < detectors_.size(); ++i) {
        detectorThreads_.emplace_back([this, i]() {
            incrementReady();
            detectors_[i]->run(featureBuffers_[i], outputMutex_, 
                              config_.outputMode, config_.showTimestamp);
        });
    }
}

void Pipeline::stop() {
    if (!running_) {
        return;
    }
    
    running_ = false;
    
    // Signal exhaustion to all buffers
    audioBuffer_->setExhausted(true);
    
    // Wait for threads to complete
    if (melThread_.joinable()) {
        melThread_.join();
    }
    
    melBuffer_->setExhausted(true);
    
    if (embeddingThread_.joinable()) {
        embeddingThread_.join();
    }
    
    for (auto& buffer : featureBuffers_) {
        buffer->setExhausted(true);
    }
    
    for (auto& thread : detectorThreads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    
    detectorThreads_.clear();
}

void Pipeline::processAudio(const AudioSample* samples, size_t sampleCount) {
    if (!running_) {
        return;
    }
    
    // Get a buffer from the pool
    auto borrowed = audioBufferPool_->borrow();
    auto& floatSamples = *borrowed;
    
    // Use SIMD-optimized conversion
    SimdAudio::convertToFloat(samples, floatSamples, sampleCount);
    
    // TODO: Apply preprocessors if needed
    // for (auto& preprocessor : preprocessors_) {
    //     if (preprocessor->isEnabled()) {
    //         preprocessor->process(floatSamples.data(), floatSamples.size());
    //     }
    // }
    
    // Push to audio buffer - move semantics will transfer ownership
    audioBuffer_->push(std::move(floatSamples));
    // borrowed object automatically returns buffer to pool when it goes out of scope
}

void Pipeline::addPreprocessor(std::unique_ptr<Preprocessor> preprocessor) {
    preprocessors_.push_back(std::move(preprocessor));
}

void Pipeline::addPostprocessor(std::unique_ptr<Postprocessor> postprocessor) {
    postprocessors_.push_back(std::move(postprocessor));
}

void Pipeline::waitUntilReady() {
    std::unique_lock<std::mutex> lock(readyMutex_);
    readyCv_.wait(lock, [this]() { 
        return readyCount_ >= expectedReadyCount_; 
    });
    
    if (config_.outputMode == OutputMode::VERBOSE || config_.outputMode == OutputMode::NORMAL) {
        std::cerr << "[LOG] Pipeline ready" << std::endl;
    }
}

void Pipeline::incrementReady() {
    std::unique_lock<std::mutex> lock(readyMutex_);
    readyCount_++;
    readyCv_.notify_one();
}

} // namespace openwakeword