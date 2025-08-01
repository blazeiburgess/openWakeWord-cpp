#include "preprocessors/vad.h"
#include <iostream>

namespace openwakeword {

VADPreprocessor::VADPreprocessor(float threshold)
    : Preprocessor("VAD"), threshold_(threshold) {
}

bool VADPreprocessor::initialize(const std::filesystem::path& modelPath,
                                Ort::Env& env,
                                const Ort::SessionOptions& options) {
    model_ = std::make_unique<VADModel>();
    if (!model_->loadModel(modelPath, env, options)) {
        std::cerr << "[ERROR] Failed to load VAD model: " << modelPath << std::endl;
        return false;
    }
    
    std::cerr << "[LOG] Loaded VAD model" << std::endl;
    return true;
}

void VADPreprocessor::process(AudioFrame& frame) {
    // Process the frame through VAD
    process(frame.samples.data(), frame.samples.size());
    
    // If voice is not detected, optionally zero out the frame
    // This is a simple approach; more sophisticated methods could be used
    if (!isVoiceDetected()) {
        // Option 1: Zero out non-voice frames
        // std::fill(frame.samples.begin(), frame.samples.end(), 0);
        
        // Option 2: Attenuate non-voice frames
        // for (auto& sample : frame.samples) {
        //     sample = static_cast<AudioSample>(sample * 0.1f);
        // }
        
        // Option 3: Do nothing, let downstream processors decide
        // This is the current approach
    }
}

void VADPreprocessor::process(AudioSample* samples, size_t count) {
    // Convert to float for VAD processing
    audioBuffer_.clear();
    audioBuffer_.reserve(count);
    
    for (size_t i = 0; i < count; ++i) {
        audioBuffer_.push_back(static_cast<AudioFloat>(samples[i]));
    }
    
    // Process through VAD model
    // Note: Silero VAD expects specific frame sizes, so we may need to buffer
    if (audioBuffer_.size() >= VAD_FRAME_SIZE) {
        lastScore_ = model_->predictVoiceActivity(audioBuffer_);
    }
}

} // namespace openwakeword