#include <cstdio>
#include <iostream>
#include <vector>
#include <csignal>
#include "core/pipeline.h"
#include "utils/config.h"

using namespace openwakeword;

// Global pipeline for signal handling
std::unique_ptr<Pipeline> g_pipeline;

void signalHandler(int signal) {
    if (signal == SIGINT || signal == SIGTERM) {
        std::cerr << "\n[LOG] Shutting down..." << std::endl;
        if (g_pipeline) {
            g_pipeline->stop();
        }
        exit(0);
    }
}

int main(int argc, char *argv[]) {
    // Install signal handlers
    std::signal(SIGINT, signalHandler);
    std::signal(SIGTERM, signalHandler);
    
    // Re-open stdin in binary mode
    std::freopen(nullptr, "rb", stdin);
    
    // Parse configuration
    Config config;
    ParseResult parseResult = config.parseArgs(argc, argv);
    if (parseResult == ParseResult::ERROR_EXIT) {
        return 1;
    } else if (parseResult == ParseResult::INFO_EXIT) {
        return 0;
    }
    
    // Create and initialize pipeline
    g_pipeline = std::make_unique<Pipeline>(config);
    
    if (!g_pipeline->initialize()) {
        std::cerr << "[ERROR] Failed to initialize pipeline" << std::endl;
        return 1;
    }
    
    // Start processing threads
    g_pipeline->start();
    
    // Wait until all components are ready
    g_pipeline->waitUntilReady();
    
    if (config.outputMode != OutputMode::QUIET) {
        std::cerr << "[LOG] Ready" << std::endl;
    }
    
    // Main audio input loop
    std::vector<AudioSample> samples(config.frameSize);
    size_t framesRead = std::fread(samples.data(), sizeof(AudioSample), 
                                   config.frameSize, stdin);
    
    while (framesRead > 0 && g_pipeline->isRunning()) {
        // Process audio through pipeline
        g_pipeline->processAudio(samples.data(), framesRead);
        
        // Read next chunk
        framesRead = std::fread(samples.data(), sizeof(AudioSample), 
                               config.frameSize, stdin);
    }
    
    // Stop pipeline
    g_pipeline->stop();
    
    return 0;
}