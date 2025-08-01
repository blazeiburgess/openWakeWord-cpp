#include "utils/config.h"
#include <iostream>
#include <cstdlib>

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
    
    // Configure ONNX Runtime options
    sessionOptions.SetIntraOpNumThreads(1);
    sessionOptions.SetInterOpNumThreads(1);
    
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
    std::cerr << "usage: " << programName << " [options]" << std::endl;
    std::cerr << std::endl;
    std::cerr << "options:" << std::endl;
    std::cerr << "   -h        --help                         show this message and exit" << std::endl;
    std::cerr << "   -m  FILE  --model FILE                   path to wake word model (repeat for multiple models)" << std::endl;
    std::cerr << "   -c  FILE  --config FILE                  path to configuration file" << std::endl;
    std::cerr << "   -t  NUM   --threshold NUM                threshold for activation (0-1, default: 0.5)" << std::endl;
    std::cerr << "   -l  NUM   --trigger-level NUM            number of activations before output (default: 4)" << std::endl;
    std::cerr << "   -r  NUM   --refractory NUM               number of steps after activation to wait (default: 20)" << std::endl;
    std::cerr << "   --step-frames NUM                        number of 80 ms audio chunks to process at a time (default: 4)" << std::endl;
    std::cerr << "   --melspectrogram-model FILE              path to melspectrogram.onnx file" << std::endl;
    std::cerr << "   --embedding-model FILE                   path to embedding_model.onnx file" << std::endl;
    std::cerr << "   --vad-threshold NUM                      enable VAD with threshold (0-1)" << std::endl;
    std::cerr << "   --vad-model FILE                         path to VAD model (default: models/silero_vad.onnx)" << std::endl;
    std::cerr << "   --enable-noise-suppression               enable Speex noise suppression" << std::endl;
    std::cerr << "   --debug                                  print model probabilities to stderr" << std::endl;
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