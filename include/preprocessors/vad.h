#ifndef OPENWAKEWORD_VAD_H
#define OPENWAKEWORD_VAD_H

#include <filesystem>
#include <memory>
#include <vector>
#include "preprocessors/preprocessor.h"
#include "core/model_wrapper.h"
#include "core/types.h"

namespace openwakeword {

// Voice Activity Detection preprocessor
class VADPreprocessor : public Preprocessor {
public:
    VADPreprocessor(float threshold = 0.5f);
    
    // Initialize VAD model
    bool initialize(const std::filesystem::path& modelPath,
                   Ort::Env& env,
                   const Ort::SessionOptions& options);
    
    // Process audio frame
    void process(AudioFrame& frame) override;
    
    // Process raw samples
    void process(AudioSample* samples, size_t count) override;
    
    // Get/set VAD threshold
    float getThreshold() const { return threshold_; }
    void setThreshold(float threshold) { threshold_ = threshold; }
    
    // Get last VAD score
    float getLastScore() const { return lastScore_; }
    
    // Check if voice is currently detected
    bool isVoiceDetected() const { return lastScore_ > threshold_; }
    
private:
    std::unique_ptr<VADModel> model_;
    float threshold_;
    float lastScore_ = 0.0f;
    std::vector<AudioFloat> audioBuffer_;
    
    // Silero VAD specific parameters
    static constexpr size_t VAD_FRAME_SIZE = 512;  // Silero VAD frame size
    static constexpr size_t VAD_SAMPLE_RATE = 16000;
};

} // namespace openwakeword

#endif // OPENWAKEWORD_VAD_H