#ifndef OPENWAKEWORD_PIPELINE_H
#define OPENWAKEWORD_PIPELINE_H

#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>
#include "core/audio_processor.h"
#include "core/types.h"
#include "processors/mel_spectrogram.h"
#include "processors/speech_embedding.h"
#include "processors/wake_word_detector.h"
#include "utils/config.h"

namespace openwakeword {

// Forward declarations
class Preprocessor;
class Postprocessor;

// Audio processing pipeline manager
class Pipeline {
public:
    Pipeline(const Config& config);
    ~Pipeline();
    
    // Initialize all components
    bool initialize();
    
    // Start processing threads
    void start();
    
    // Stop processing threads
    void stop();
    
    // Process audio input
    void processAudio(const AudioSample* samples, size_t sampleCount);
    
    // Add preprocessor to pipeline
    void addPreprocessor(std::unique_ptr<Preprocessor> preprocessor);
    
    // Add postprocessor to pipeline  
    void addPostprocessor(std::unique_ptr<Postprocessor> postprocessor);
    
    // Check if pipeline is running
    bool isRunning() const { return running_; }
    
    // Wait until pipeline is ready
    void waitUntilReady();
    
private:
    Config config_;
    Ort::Env env_;
    Ort::SessionOptions sessionOptions_;
    
    // Processing stages
    std::unique_ptr<MelSpectrogramProcessor> melProcessor_;
    std::unique_ptr<SpeechEmbeddingProcessor> embeddingProcessor_;
    std::vector<std::unique_ptr<WakeWordDetector>> detectors_;
    
    // Preprocessors and postprocessors
    std::vector<std::unique_ptr<Preprocessor>> preprocessors_;
    std::vector<std::unique_ptr<Postprocessor>> postprocessors_;
    
    // Buffers for inter-processor communication
    std::shared_ptr<ThreadSafeBuffer<AudioFloat>> audioBuffer_;
    std::shared_ptr<ThreadSafeBuffer<AudioFloat>> melBuffer_;
    std::vector<std::shared_ptr<ThreadSafeBuffer<AudioFloat>>> featureBuffers_;
    
    // Processing threads
    std::thread melThread_;
    std::thread embeddingThread_;
    std::vector<std::thread> detectorThreads_;
    
    // Synchronization
    std::mutex outputMutex_;
    std::mutex readyMutex_;
    std::condition_variable readyCv_;
    size_t readyCount_ = 0;
    size_t expectedReadyCount_ = 0;
    bool running_ = false;
    
    // Helper methods
    void runAudioInput();
    void incrementReady();
};

} // namespace openwakeword

#endif // OPENWAKEWORD_PIPELINE_H