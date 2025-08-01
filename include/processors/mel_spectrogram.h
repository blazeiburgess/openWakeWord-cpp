#ifndef OPENWAKEWORD_MEL_SPECTROGRAM_H
#define OPENWAKEWORD_MEL_SPECTROGRAM_H

#include <filesystem>
#include <memory>
#include <vector>
#include "core/audio_processor.h"
#include "core/model_wrapper.h"
#include "core/types.h"
#include "core/thread_safe_buffer.h"
#include "utils/config.h"

namespace openwakeword {

// Mel spectrogram processor
class MelSpectrogramProcessor : public TransformProcessor<AudioFloat, AudioFloat> {
public:
    MelSpectrogramProcessor(Ort::Env& env, const Ort::SessionOptions& options);
    
    // Set model path
    void setModelPath(const std::filesystem::path& path) { modelPath_ = path; }
    
    // Set frame size
    void setFrameSize(size_t frameSize) { frameSize_ = frameSize; }
    
    // AudioProcessor interface
    bool initialize() override;
    bool process() override;
    void reset() override;
    
    // Thread entry point
    void run(std::shared_ptr<ThreadSafeBuffer<AudioFloat>> input,
             std::shared_ptr<ThreadSafeBuffer<AudioFloat>> output,
             OutputMode outputMode);
    
private:
    Ort::Env& env_;
    const Ort::SessionOptions& options_;
    std::filesystem::path modelPath_;
    size_t frameSize_ = 4 * CHUNK_SAMPLES;
    
    std::unique_ptr<MelSpectrogramModel> model_;
    std::vector<AudioFloat> todoSamples_;
};

} // namespace openwakeword

#endif // OPENWAKEWORD_MEL_SPECTROGRAM_H