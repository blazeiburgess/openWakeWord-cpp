#ifndef OPENWAKEWORD_CONFIG_H
#define OPENWAKEWORD_CONFIG_H

#include <filesystem>
#include <string>
#include <vector>
#include <onnxruntime_cxx_api.h>
#include "core/types.h"

namespace openwakeword {

// Forward declaration
struct WakeWordConfig;

// Version information
constexpr const char* VERSION = "1.0.0";
constexpr const char* BUILD_DATE = __DATE__;
constexpr const char* BUILD_TIME = __TIME__;

// Output modes
enum class OutputMode {
    NORMAL,
    QUIET,
    VERBOSE,
    JSON
};

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
    
    // Output control
    OutputMode outputMode = OutputMode::NORMAL;
    bool showTimestamp = false;
    bool jsonOutput = false;
    
    // Advanced options
    bool enableCustomVerifiers = false;
    float customVerifierThreshold = 0.1f;
    
    // ONNX Runtime configuration
    int intraOpNumThreads = 1;
    int interOpNumThreads = 1;
    
    // Parse command line arguments
    bool parseArgs(int argc, char* argv[]);
    
    // Load from configuration file
    bool loadFromFile(const std::filesystem::path& configPath);
    
    // Validate configuration
    bool validate() const;
    
    // Print usage
    static void printUsage(const char* programName);
    
    // Print version information
    static void printVersion();
    
    // List available models
    static void listAvailableModels();
    
    // Save configuration to file
    bool saveToFile(const std::filesystem::path& configPath) const;
    
private:
    // Helper to ensure argument exists
    static bool ensureArg(int argc, char* argv[], int& index);
};

} // namespace openwakeword

#endif // OPENWAKEWORD_CONFIG_H