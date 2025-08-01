#include "preprocessors/speex_noise_suppressor.h"
#include <iostream>
#include <cstring>

#ifdef HAVE_SPEEX
#include <speex/speex_preprocess.h>
#endif

namespace openwakeword {

SpeexNoiseSupressor::SpeexNoiseSupressor(int sampleRate, int frameSize)
    : Preprocessor("SpeexNoiseSuppression"), 
      frameSize_(frameSize), 
      sampleRate_(sampleRate) {
#ifdef HAVE_SPEEX
    state_ = speex_preprocess_state_init(frameSize_, sampleRate_);
    if (state_) {
        // Enable noise suppression by default
        int denoise = 1;
        speex_preprocess_ctl(state_, SPEEX_PREPROCESS_SET_DENOISE, &denoise);
        
        // Set default noise suppression level
        int noiseSuppress = -25;
        speex_preprocess_ctl(state_, SPEEX_PREPROCESS_SET_NOISE_SUPPRESS, &noiseSuppress);
        
        std::cerr << "[LOG] Initialized Speex noise suppression" << std::endl;
    }
#else
    std::cerr << "[WARNING] Speex support not compiled in" << std::endl;
    setEnabled(false);
#endif
}

SpeexNoiseSupressor::~SpeexNoiseSupressor() {
#ifdef HAVE_SPEEX
    if (state_) {
        speex_preprocess_state_destroy(state_);
    }
#endif
}

void SpeexNoiseSupressor::process(AudioFrame& frame) {
    process(frame.samples.data(), frame.samples.size());
}

void SpeexNoiseSupressor::process(AudioSample* samples, size_t count) {
#ifdef HAVE_SPEEX
    if (!state_ || !isEnabled()) {
        return;
    }
    
    // Process in frame-sized chunks
    size_t processed = 0;
    
    // Ensure process buffer is large enough
    if (processBuffer_.size() < static_cast<size_t>(frameSize_)) {
        processBuffer_.resize(frameSize_);
    }
    
    while (processed + static_cast<size_t>(frameSize_) <= count) {
        // Copy to process buffer
        std::memcpy(processBuffer_.data(), samples + processed, 
                   frameSize_ * sizeof(AudioSample));
        
        // Process through Speex
        speex_preprocess_run(state_, processBuffer_.data());
        
        // Copy back
        std::memcpy(samples + processed, processBuffer_.data(), 
                   frameSize_ * sizeof(AudioSample));
        
        processed += static_cast<size_t>(frameSize_);
    }
    
    // Handle remaining samples (if any)
    if (processed < count) {
        size_t remaining = count - processed;
        
        // Zero-pad the buffer
        std::memcpy(processBuffer_.data(), samples + processed, 
                   remaining * sizeof(AudioSample));
        std::memset(processBuffer_.data() + remaining, 0, 
                   (frameSize_ - remaining) * sizeof(AudioSample));
        
        // Process
        speex_preprocess_run(state_, processBuffer_.data());
        
        // Copy back only the valid samples
        std::memcpy(samples + processed, processBuffer_.data(), 
                   remaining * sizeof(AudioSample));
    }
#endif
}

void SpeexNoiseSupressor::setSuppressionLevel(int level) {
#ifdef HAVE_SPEEX
    if (state_) {
        speex_preprocess_ctl(state_, SPEEX_PREPROCESS_SET_NOISE_SUPPRESS, &level);
    }
#endif
}

void SpeexNoiseSupressor::setVAD(bool enable) {
#ifdef HAVE_SPEEX
    if (state_) {
        int vad = enable ? 1 : 0;
        speex_preprocess_ctl(state_, SPEEX_PREPROCESS_SET_VAD, &vad);
    }
#endif
}

void SpeexNoiseSupressor::setDenoise(bool enable) {
#ifdef HAVE_SPEEX
    if (state_) {
        int denoise = enable ? 1 : 0;
        speex_preprocess_ctl(state_, SPEEX_PREPROCESS_SET_DENOISE, &denoise);
    }
#endif
}

bool SpeexNoiseSupressor::isAvailable() {
#ifdef HAVE_SPEEX
    return true;
#else
    return false;
#endif
}

bool SpeexNoiseSupressor::checkAvailability() {
    return isAvailable();
}

} // namespace openwakeword