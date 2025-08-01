#ifndef OPENWAKEWORD_SPEEX_NOISE_SUPPRESSOR_H
#define OPENWAKEWORD_SPEEX_NOISE_SUPPRESSOR_H

#include <memory>
#include <vector>
#include "preprocessors/preprocessor.h"
#include "core/types.h"

// Forward declaration to avoid including Speex headers
struct SpeexPreprocessState_;
typedef struct SpeexPreprocessState_ SpeexPreprocessState;

namespace openwakeword {

// Speex noise suppression preprocessor
class SpeexNoiseSupressor : public Preprocessor {
public:
    SpeexNoiseSupressor(int sampleRate = SAMPLE_RATE, int frameSize = 320);
    ~SpeexNoiseSupressor();
    
    // Process audio frame
    void process(AudioFrame& frame) override;
    
    // Process raw samples
    void process(AudioSample* samples, size_t count) override;
    
    // Set noise suppression level (-30 to -0 dB)
    void setSuppressionLevel(int level);
    
    // Set VAD option for Speex
    void setVAD(bool enable);
    
    // Set denoise option
    void setDenoise(bool enable);
    
    // Check if Speex is available
    static bool isAvailable();
    
private:
    SpeexPreprocessState* state_ = nullptr;
    int frameSize_;
    int sampleRate_;
    std::vector<AudioSample> processBuffer_;
    
    // Speex availability flag
    static bool checkAvailability();
};

} // namespace openwakeword

#endif // OPENWAKEWORD_SPEEX_NOISE_SUPPRESSOR_H