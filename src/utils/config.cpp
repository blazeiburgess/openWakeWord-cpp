#include "utils/config.h"
#include <iostream>
#include <cstdlib>
#include <filesystem>
#include <algorithm>
#include <iomanip>
#include <fstream>
#include <onnxruntime_cxx_api.h>

namespace openwakeword {

bool Config::parseArgs(int argc, char* argv[]) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "-m" || arg == "--model") {
            if (!ensureArg(argc, argv, i)) return false;
            wakeWordModelPaths.push_back(argv[++i]);
        } else if (arg == "-t" || arg == "--threshold") {
            if (!ensureArg(argc, argv, i)) return false;
            threshold = std::atof(argv[++i]);
        } else if (arg == "-l" || arg == "--trigger-level") {
            if (!ensureArg(argc, argv, i)) return false;
            triggerLevel = std::atoi(argv[++i]);
        } else if (arg == "-r" || arg == "--refractory") {
            if (!ensureArg(argc, argv, i)) return false;
            refractorySteps = std::atoi(argv[++i]);
        } else if (arg == "--step-frames") {
            if (!ensureArg(argc, argv, i)) return false;
            stepFrames = std::atoi(argv[++i]);
        } else if (arg == "--melspectrogram-model") {
            if (!ensureArg(argc, argv, i)) return false;
            melModelPath = argv[++i];
        } else if (arg == "--embedding-model") {
            if (!ensureArg(argc, argv, i)) return false;
            embModelPath = argv[++i];
        } else if (arg == "--vad-threshold") {
            if (!ensureArg(argc, argv, i)) return false;
            vadThreshold = std::atof(argv[++i]);
            enableVAD = true;
        } else if (arg == "--vad-model") {
            if (!ensureArg(argc, argv, i)) return false;
            vadModelPath = argv[++i];
            enableVAD = true;
        } else if (arg == "--enable-noise-suppression") {
            enableNoiseSuppression = true;
        } else if (arg == "--debug") {
            debug = true;
        } else if (arg == "--quiet") {
            outputMode = OutputMode::QUIET;
        } else if (arg == "--verbose") {
            outputMode = OutputMode::VERBOSE;
        } else if (arg == "--json") {
            jsonOutput = true;
            outputMode = OutputMode::JSON;
        } else if (arg == "--timestamp") {
            showTimestamp = true;
        } else if (arg == "--version") {
            printVersion();
            return false;
        } else if (arg == "--list-models") {
            listAvailableModels();
            return false;
        } else if (arg == "-h" || arg == "--help") {
            printUsage(argv[0]);
            return false;
        } else if (arg == "-c" || arg == "--config") {
            if (!ensureArg(argc, argv, i)) return false;
            if (!loadFromFile(argv[++i])) {
                return false;
            }
        } else {
            std::cerr << "[ERROR] Unknown argument: " << arg << std::endl;
            printUsage(argv[0]);
            return false;
        }
    }
    
    // Update frame size based on step frames
    frameSize = stepFrames * CHUNK_SAMPLES;
    
    // Configure ONNX Runtime options (these will be used when creating SessionOptions)
    // intraOpNumThreads = 1;  // Already set as default
    // interOpNumThreads = 1;  // Already set as default
    
    // Create per-model configurations if not already specified
    if (wakeWordConfigs.empty() && !wakeWordModelPaths.empty()) {
        for (const auto& modelPath : wakeWordModelPaths) {
            WakeWordConfig config;
            config.modelPath = modelPath;
            config.threshold = threshold;
            config.triggerLevel = triggerLevel;
            config.refractorySteps = refractorySteps;
            config.debug = debug;
            wakeWordConfigs.push_back(config);
        }
    }
    
    return validate();
}

bool Config::loadFromFile(const std::filesystem::path& configPath) {
    // TODO: Implement JSON/YAML configuration file loading
    (void)configPath; // Suppress unused parameter warning
    std::cerr << "[WARNING] Configuration file loading not yet implemented" << std::endl;
    return true;
}

bool Config::validate() const {
    if (wakeWordModelPaths.empty() && wakeWordConfigs.empty()) {
        std::cerr << "[ERROR] No wake word models specified" << std::endl;
        return false;
    }
    
    if (!std::filesystem::exists(melModelPath)) {
        std::cerr << "[ERROR] Mel spectrogram model not found: " << melModelPath << std::endl;
        return false;
    }
    
    if (!std::filesystem::exists(embModelPath)) {
        std::cerr << "[ERROR] Embedding model not found: " << embModelPath << std::endl;
        return false;
    }
    
    if (enableVAD && !std::filesystem::exists(vadModelPath)) {
        std::cerr << "[ERROR] VAD model not found: " << vadModelPath << std::endl;
        return false;
    }
    
    for (const auto& modelPath : wakeWordModelPaths) {
        if (!std::filesystem::exists(modelPath)) {
            std::cerr << "[ERROR] Wake word model not found: " << modelPath << std::endl;
            return false;
        }
    }
    
    if (threshold < 0.0f || threshold > 1.0f) {
        std::cerr << "[ERROR] Threshold must be between 0 and 1" << std::endl;
        return false;
    }
    
    if (vadThreshold < 0.0f || vadThreshold > 1.0f) {
        std::cerr << "[ERROR] VAD threshold must be between 0 and 1" << std::endl;
        return false;
    }
    
    return true;
}

void Config::printUsage(const char* programName) {
    std::cerr << std::endl;
    std::cerr << "openWakeWord - Real-time wake word detection" << std::endl;
    std::cerr << std::endl;
    std::cerr << "USAGE:" << std::endl;
    std::cerr << "  " << programName << " [options]" << std::endl;
    std::cerr << std::endl;
    std::cerr << "GENERAL OPTIONS:" << std::endl;
    std::cerr << "  -h, --help                    Show this help message and exit" << std::endl;
    std::cerr << "  --version                     Show version information" << std::endl;
    std::cerr << "  --list-models                 List available wake word models" << std::endl;
    std::cerr << "  -c, --config FILE             Load configuration from file" << std::endl;
    std::cerr << std::endl;
    std::cerr << "MODEL OPTIONS:" << std::endl;
    std::cerr << "  -m, --model FILE              Path to wake word model (can be repeated)" << std::endl;
    std::cerr << "  -t, --threshold NUM           Detection threshold (0-1, default: 0.5)" << std::endl;
    std::cerr << "  -l, --trigger-level NUM       Activations needed before trigger (default: 4)" << std::endl;
    std::cerr << "  -r, --refractory NUM          Steps to wait after activation (default: 20)" << std::endl;
    std::cerr << "  --melspectrogram-model FILE   Path to mel spectrogram model" << std::endl;
    std::cerr << "  --embedding-model FILE        Path to speech embedding model" << std::endl;
    std::cerr << std::endl;
    std::cerr << "AUDIO PROCESSING:" << std::endl;
    std::cerr << "  --enable-noise-suppression    Enable Speex noise suppression" << std::endl;
    std::cerr << "  --vad-threshold NUM           Enable VAD with threshold (0-1)" << std::endl;
    std::cerr << "  --vad-model FILE              Path to VAD model" << std::endl;
    std::cerr << "  --step-frames NUM             Audio chunks to process at once (default: 4)" << std::endl;
    std::cerr << std::endl;
    std::cerr << "OUTPUT OPTIONS:" << std::endl;
    std::cerr << "  --quiet                       Suppress all output except detections" << std::endl;
    std::cerr << "  --verbose                     Enable verbose logging" << std::endl;
    std::cerr << "  --json                        Output in JSON format" << std::endl;
    std::cerr << "  --timestamp                   Include timestamps with detections" << std::endl;
    std::cerr << "  --debug                       Print model probabilities to stderr" << std::endl;
    std::cerr << std::endl;
    std::cerr << "EXAMPLES:" << std::endl;
    std::cerr << "  # Basic usage with single model" << std::endl;
    std::cerr << "  arecord -r 16000 -c 1 -f S16_LE -t raw - | " << programName << " --model models/alexa_v0.1.onnx" << std::endl;
    std::cerr << std::endl;
    std::cerr << "  # Multiple models with noise suppression" << std::endl;
    std::cerr << "  arecord -r 16000 -c 1 -f S16_LE -t raw - | " << programName << " \\" << std::endl;
    std::cerr << "    --model models/alexa_v0.1.onnx --model models/hey_jarvis_v0.1.onnx \\" << std::endl;
    std::cerr << "    --enable-noise-suppression --threshold 0.6" << std::endl;
    std::cerr << std::endl;
}

bool Config::ensureArg(int argc, char* argv[], int& index) {
    if ((index + 1) >= argc) {
        std::cerr << "[ERROR] Missing value for argument: " << argv[index] << std::endl;
        printUsage(argv[0]);
        return false;
    }
    return true;
}

} // namespace openwakeword