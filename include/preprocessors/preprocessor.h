#ifndef OPENWAKEWORD_PREPROCESSOR_H
#define OPENWAKEWORD_PREPROCESSOR_H

#include <string>
#include "core/types.h"

namespace openwakeword {

// Base interface for audio preprocessors
class Preprocessor {
public:
    Preprocessor(const std::string& name) : name_(name) {}
    virtual ~Preprocessor() = default;
    
    // Process audio frame in-place
    virtual void process(AudioFrame& frame) = 0;
    
    // Process raw samples
    virtual void process(AudioSample* samples, size_t count) = 0;
    
    // Check if preprocessor is enabled
    virtual bool isEnabled() const { return enabled_; }
    virtual void setEnabled(bool enabled) { enabled_ = enabled; }
    
    // Get preprocessor name
    const std::string& getName() const { return name_; }
    
protected:
    std::string name_;
    bool enabled_ = true;
};

} // namespace openwakeword

#endif // OPENWAKEWORD_PREPROCESSOR_H