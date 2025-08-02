#include "processors/mel_spectrogram.h"
#include <iostream>

namespace openwakeword {

MelSpectrogramProcessor::MelSpectrogramProcessor(Ort::Env& env, const Ort::SessionOptions& options)
    : TransformProcessor("MelSpectrogram"), 
      env_(env), 
      options_(options),
      audioBuffer_(16 * CHUNK_SAMPLES) {  // Buffer for up to 16 chunks
}

bool MelSpectrogramProcessor::initialize() {
    if (!std::filesystem::exists(modelPath_)) {
        std::cerr << "[ERROR] Mel spectrogram model not found: " << modelPath_ << std::endl;
        return false;
    }
    
    model_ = std::make_unique<MelSpectrogramModel>();
    if (!model_->loadModel(modelPath_, env_, options_)) {
        std::cerr << "[ERROR] Failed to load mel spectrogram model" << std::endl;
        return false;
    }
    
    // Log message handled by pipeline based on output mode
    initialized_ = true;
    return true;
}

bool MelSpectrogramProcessor::process() {
    // This method is not used in threaded mode
    return true;
}

void MelSpectrogramProcessor::reset() {
    audioBuffer_.clear();
}

void MelSpectrogramProcessor::run(std::shared_ptr<ThreadSafeBuffer<AudioFloat>> input,
                                  std::shared_ptr<ThreadSafeBuffer<AudioFloat>> output,
                                  OutputMode outputMode) {
    if (!initialized_) {
        if (outputMode != OutputMode::QUIET) {
            std::cerr << "[ERROR] MelSpectrogramProcessor not initialized" << std::endl;
        }
        return;
    }
    
    AudioBuffer frameSamples(frameSize_);
    
    while (true) {
        // Get audio samples from input buffer
        auto samples = input->pull();
        if (input->isExhausted() && samples.empty()) {
            break;
        }
        
        // Push samples to ring buffer
        if (!samples.empty()) {
            audioBuffer_.push(samples);
        }
        
        // Process complete frames
        while (audioBuffer_.size() >= frameSize_) {
            // Pop one frame directly into our pre-allocated buffer
            audioBuffer_.pop(frameSamples, frameSize_);
            
            // Compute mel spectrogram
            auto melData = model_->computeMelSpectrogram(frameSamples);
            
            // Push to output buffer
            output->push(melData);
        }
    }
    
    // Signal that processing is complete
    output->setExhausted(true);
}

} // namespace openwakeword