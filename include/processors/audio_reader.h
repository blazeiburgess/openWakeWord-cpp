#ifndef OPENWAKEWORD_AUDIO_READER_H
#define OPENWAKEWORD_AUDIO_READER_H

#include <cstdio>
#include <memory>
#include <vector>
#include "core/types.h"

namespace openwakeword {

// Interface for audio input sources
class AudioReader {
public:
    virtual ~AudioReader() = default;
    
    // Read audio samples
    virtual size_t read(AudioSample* buffer, size_t samples) = 0;
    
    // Check if more audio is available
    virtual bool hasMore() const = 0;
    
    // Get sample rate
    virtual size_t getSampleRate() const { return SAMPLE_RATE; }
};

// Stdin audio reader
class StdinAudioReader : public AudioReader {
public:
    StdinAudioReader();
    
    size_t read(AudioSample* buffer, size_t samples) override;
    bool hasMore() const override;
    
private:
    FILE* input_;
    bool eof_ = false;
};

// WAV file audio reader (for future batch processing)
class WavFileReader : public AudioReader {
public:
    explicit WavFileReader(const std::string& filename);
    ~WavFileReader();
    
    size_t read(AudioSample* buffer, size_t samples) override;
    bool hasMore() const override;
    size_t getSampleRate() const override { return sampleRate_; }
    
private:
    FILE* file_ = nullptr;
    size_t sampleRate_ = SAMPLE_RATE;
    size_t remainingSamples_ = 0;
    bool headerParsed_ = false;
    
    bool parseHeader();
};

} // namespace openwakeword

#endif // OPENWAKEWORD_AUDIO_READER_H