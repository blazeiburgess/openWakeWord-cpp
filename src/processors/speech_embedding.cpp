#include "processors/speech_embedding.h"
#include <iostream>

namespace openwakeword {

SpeechEmbeddingProcessor::SpeechEmbeddingProcessor(Ort::Env& env, 
                                                   const Ort::SessionOptions& options,
                                                   size_t numWakeWords)
    : TransformProcessor("SpeechEmbedding"), 
      env_(env), 
      options_(options), 
      numWakeWords_(numWakeWords),
      melBuffer_(EMBEDDING_WINDOW_SIZE * NUM_MELS * 2) {  // Buffer for 2 windows worth of mels
}

bool SpeechEmbeddingProcessor::initialize() {
    if (!std::filesystem::exists(modelPath_)) {
        std::cerr << "[ERROR] Speech embedding model not found: " << modelPath_ << std::endl;
        return false;
    }
    
    model_ = std::make_unique<EmbeddingModel>();
    if (!model_->loadModel(modelPath_, env_, options_)) {
        std::cerr << "[ERROR] Failed to load speech embedding model" << std::endl;
        return false;
    }
    
    // Log message handled by pipeline based on output mode
    initialized_ = true;
    return true;
}

bool SpeechEmbeddingProcessor::process() {
    // This method is not used in threaded mode
    return true;
}

void SpeechEmbeddingProcessor::reset() {
    melBuffer_.clear();
}

void SpeechEmbeddingProcessor::run(std::shared_ptr<ThreadSafeBuffer<AudioFloat>> input,
                                   std::vector<std::shared_ptr<ThreadSafeBuffer<AudioFloat>>> outputs,
                                   OutputMode outputMode) {
    if (!initialized_) {
        if (outputMode != OutputMode::QUIET) {
            std::cerr << "[ERROR] SpeechEmbeddingProcessor not initialized" << std::endl;
        }
        return;
    }
    
    MelBuffer windowMels(EMBEDDING_WINDOW_SIZE * NUM_MELS);
    
    while (true) {
        // Get mel spectrograms from input buffer
        auto mels = input->pull();
        if (input->isExhausted() && mels.empty()) {
            break;
        }
        
        // Push mels to ring buffer
        if (!mels.empty()) {
            melBuffer_.push(mels);
        }
        
        // Process when we have enough mel frames
        size_t melFrames = melBuffer_.size() / NUM_MELS;
        while (melFrames >= EMBEDDING_WINDOW_SIZE) {
            // Peek window of mels (don't remove yet for sliding window)
            melBuffer_.peek(windowMels.data(), EMBEDDING_WINDOW_SIZE * NUM_MELS);
            
            // Extract embeddings
            auto embeddings = model_->extractEmbeddings(windowMels);
            
            // Send to all wake word detectors
            for (auto& output : outputs) {
                output->push(embeddings);
            }
            
            // Slide window by step size
            melBuffer_.skip(EMBEDDING_STEP_SIZE * NUM_MELS);
            
            melFrames = melBuffer_.size() / NUM_MELS;
        }
    }
    
    // Signal that processing is complete for all outputs
    for (auto& output : outputs) {
        output->setExhausted(true);
    }
}

} // namespace openwakeword