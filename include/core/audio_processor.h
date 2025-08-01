#ifndef OPENWAKEWORD_AUDIO_PROCESSOR_H
#define OPENWAKEWORD_AUDIO_PROCESSOR_H

#include <memory>
#include <string>
#include <vector>
#include "types.h"

namespace openwakeword {

// Forward declarations
class ProcessorState;

// Base class for all audio processing stages
class AudioProcessor {
public:
    AudioProcessor(const std::string& name) : processorName_(name) {}
    virtual ~AudioProcessor() = default;
    
    // Initialize the processor
    virtual bool initialize() = 0;
    
    // Process audio data (to be implemented by derived classes)
    virtual bool process() = 0;
    
    // Reset processor state
    virtual void reset() { /* Default implementation */ }
    
    // Get processor name
    const std::string& getName() const { return processorName_; }
    
    // Check if processor is ready
    virtual bool isReady() const { return initialized_; }
    
protected:
    std::string processorName_;
    bool initialized_ = false;
};

// Base class for audio processors that transform audio data
template<typename InputType, typename OutputType>
class TransformProcessor : public AudioProcessor {
public:
    TransformProcessor(const std::string& name) : AudioProcessor(name) {}
    
    // Set input/output buffers
    void setInputBuffer(std::shared_ptr<std::vector<InputType>> input) {
        inputBuffer_ = input;
    }
    
    void setOutputBuffer(std::shared_ptr<std::vector<OutputType>> output) {
        outputBuffer_ = output;
    }
    
protected:
    std::shared_ptr<std::vector<InputType>> inputBuffer_;
    std::shared_ptr<std::vector<OutputType>> outputBuffer_;
};

// Interface for preprocessors (noise suppression, VAD, etc.)
class Preprocessor {
public:
    virtual ~Preprocessor() = default;
    
    // Process audio frame in-place
    virtual void process(AudioFrame& frame) = 0;
    
    // Check if preprocessor is enabled
    virtual bool isEnabled() const = 0;
    
    // Get preprocessor name
    virtual const std::string& getName() const = 0;
};

// Interface for postprocessors (custom verifiers, etc.)
class Postprocessor {
public:
    virtual ~Postprocessor() = default;
    
    // Process detection result
    virtual Detection process(const Detection& detection, const FeatureBuffer& features) = 0;
    
    // Check if postprocessor should be applied to this detection
    virtual bool shouldProcess(const Detection& detection) const = 0;
    
    // Get postprocessor name
    virtual const std::string& getName() const = 0;
};

} // namespace openwakeword

#endif // OPENWAKEWORD_AUDIO_PROCESSOR_H