#ifndef OPENWAKEWORD_CONFIG_H
#define OPENWAKEWORD_CONFIG_H

#include <filesystem>
#include <string>
#include <vector>
#include <onnxruntime_cxx_api.h>
#include "processors/wake_word_detector.h"

namespace openwakeword {

// Main configuration structure
struct Config {
    // Model paths
    std::filesystem::path melModelPath = "models/melspectrogram.onnx";
    std::filesystem::path embModelPath = "models/embedding_model.onnx";
    std::vector<std::filesystem::path> wakeWordModelPaths;
    
    // Processing parameters
    size_t frameSize = 4 * CHUNK_SAMPLES;
    size_t stepFrames = 4;
    
    // Detection parameters (default for all models)
    float threshold = 0.5f;
    int triggerLevel = 4;
    int refractorySteps = 20;
    
    // Per-model configurations
    std::vector<WakeWordConfig> wakeWordConfigs;
    
    // Feature flags
    bool debug = false;
    bool enableVAD = false;
    float vadThreshold = 0.5f;
    std::filesystem::path vadModelPath = "models/silero_vad.onnx";
    
    bool enableNoiseSuppression = false;
    
    // Advanced options
    bool enableCustomVerifiers = false;
    float customVerifierThreshold = 0.1f;
    
    // ONNX Runtime options
    Ort::SessionOptions sessionOptions;
    
    // Parse command line arguments
    bool parseArgs(int argc, char* argv[]);
    
    // Load from configuration file
    bool loadFromFile(const std::filesystem::path& configPath);
    
    // Validate configuration
    bool validate() const;
    
    // Print usage
    static void printUsage(const char* programName);
    
private:
    // Helper to ensure argument exists
    static bool ensureArg(int argc, char* argv[], int& index);
};

} // namespace openwakeword

#endif // OPENWAKEWORD_CONFIG_H