#include "processors/mel_spectrogram.h"
#include <iostream>

namespace openwakeword {

MelSpectrogramProcessor::MelSpectrogramProcessor(Ort::Env& env, const Ort::SessionOptions& options)
    : TransformProcessor("MelSpectrogram"), env_(env), options_(options) {
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
    
    std::cerr << "[LOG] Loaded mel spectrogram model" << std::endl;
    initialized_ = true;
    return true;
}

bool MelSpectrogramProcessor::process() {
    // This method is not used in threaded mode
    return true;
}

void MelSpectrogramProcessor::reset() {
    todoSamples_.clear();
}

void MelSpectrogramProcessor::run(std::shared_ptr<ThreadSafeBuffer<AudioFloat>> input,
                                  std::shared_ptr<ThreadSafeBuffer<AudioFloat>> output) {
    if (!initialized_) {
        std::cerr << "[ERROR] MelSpectrogramProcessor not initialized" << std::endl;
        return;
    }
    
    while (true) {
        // Get audio samples from input buffer
        auto samples = input->pull();
        if (input->isExhausted() && samples.empty()) {
            break;
        }
        
        // Accumulate samples
        todoSamples_.insert(todoSamples_.end(), samples.begin(), samples.end());
        
        // Process complete frames
        while (todoSamples_.size() >= frameSize_) {
            // Extract one frame
            AudioBuffer frameSamples(todoSamples_.begin(), todoSamples_.begin() + frameSize_);
            
            // Compute mel spectrogram
            auto melData = model_->computeMelSpectrogram(frameSamples);
            
            // Push to output buffer
            output->push(melData);
            
            // Remove processed samples
            todoSamples_.erase(todoSamples_.begin(), todoSamples_.begin() + frameSize_);
        }
    }
    
    // Signal that processing is complete
    output->setExhausted(true);
}

} // namespace openwakeword