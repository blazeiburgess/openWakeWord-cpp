#ifndef OPENWAKEWORD_SPEECH_EMBEDDING_H
#define OPENWAKEWORD_SPEECH_EMBEDDING_H

#include <filesystem>
#include <memory>
#include <vector>
#include "core/audio_processor.h"
#include "core/model_wrapper.h"
#include "core/types.h"
#include "processors/mel_spectrogram.h"
#include "utils/config.h"

namespace openwakeword {

// Speech embedding processor
class SpeechEmbeddingProcessor : public TransformProcessor<AudioFloat, AudioFloat> {
public:
    SpeechEmbeddingProcessor(Ort::Env& env, const Ort::SessionOptions& options, 
                            size_t numWakeWords);
    
    // Set model path
    void setModelPath(const std::filesystem::path& path) { modelPath_ = path; }
    
    // AudioProcessor interface
    bool initialize() override;
    bool process() override;
    void reset() override;
    
    // Thread entry point
    void run(std::shared_ptr<ThreadSafeBuffer<AudioFloat>> input,
             std::vector<std::shared_ptr<ThreadSafeBuffer<AudioFloat>>> outputs,
             OutputMode outputMode);
    
private:
    Ort::Env& env_;
    const Ort::SessionOptions& options_;
    std::filesystem::path modelPath_;
    size_t numWakeWords_;
    
    std::unique_ptr<EmbeddingModel> model_;
    std::vector<AudioFloat> todoMels_;
};

} // namespace openwakeword

#endif // OPENWAKEWORD_SPEECH_EMBEDDING_H