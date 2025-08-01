#ifndef OPENWAKEWORD_WAKE_WORD_DETECTOR_H
#define OPENWAKEWORD_WAKE_WORD_DETECTOR_H

#include <filesystem>
#include <memory>
#include <mutex>
#include <vector>
#include "core/audio_processor.h"
#include "core/model_wrapper.h"
#include "core/types.h"
#include "utils/config.h"

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
    std::vector<AudioFloat> todoFeatures_;
    
    // Activation tracking
    int activationCount_ = 0;
    
    // Process a single prediction
    void processPrediction(float probability, std::mutex& outputMutex,
                          OutputMode outputMode, bool showTimestamp);
};

} // namespace openwakeword

#endif // OPENWAKEWORD_WAKE_WORD_DETECTOR_H