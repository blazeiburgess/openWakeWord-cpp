#ifndef OPENWAKEWORD_TYPES_H
#define OPENWAKEWORD_TYPES_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace openwakeword {

// Audio processing constants
constexpr size_t SAMPLE_RATE = 16000;
constexpr size_t CHUNK_SAMPLES = 1280;  // 80 ms
constexpr size_t NUM_MELS = 32;
constexpr size_t EMBEDDING_WINDOW_SIZE = 76;  // 775 ms
constexpr size_t EMBEDDING_STEP_SIZE = 8;     // 80 ms
constexpr size_t EMBEDDING_FEATURES = 96;
constexpr size_t WAKEWORD_FEATURES = 16;

// Type aliases for clarity
using AudioSample = int16_t;
using AudioFloat = float;
using AudioBuffer = std::vector<AudioFloat>;
using MelBuffer = std::vector<AudioFloat>;
using FeatureBuffer = std::vector<AudioFloat>;

// Detection result
struct Detection {
    std::string modelName;
    float score;
    size_t frameIndex;
    
    Detection(const std::string& name, float s, size_t idx) 
        : modelName(name), score(s), frameIndex(idx) {}
};

// Model types
enum class ModelType {
    MELSPECTROGRAM,
    EMBEDDING,
    WAKEWORD,
    VAD,
    CUSTOM_VERIFIER
};

// Audio frame for processing
struct AudioFrame {
    std::vector<AudioSample> samples;
    size_t sampleRate;
    size_t timestamp;  // in samples
    
    AudioFrame() : sampleRate(SAMPLE_RATE), timestamp(0) {}
    
    size_t size() const { return samples.size(); }
    bool empty() const { return samples.empty(); }
    void clear() { samples.clear(); }
};

} // namespace openwakeword

#endif // OPENWAKEWORD_TYPES_H