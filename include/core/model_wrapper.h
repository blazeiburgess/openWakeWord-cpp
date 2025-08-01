#ifndef OPENWAKEWORD_MODEL_WRAPPER_H
#define OPENWAKEWORD_MODEL_WRAPPER_H

#include <filesystem>
#include <memory>
#include <string>
#include <vector>
#include <onnxruntime_cxx_api.h>
#include "types.h"

namespace openwakeword {

// Base class for ONNX model wrappers
class ModelWrapper {
public:
    ModelWrapper(const std::string& modelName, ModelType type);
    virtual ~ModelWrapper() = default;
    
    // Load model from file
    bool loadModel(const std::filesystem::path& modelPath, 
                   Ort::Env& env,
                   const Ort::SessionOptions& options);
    
    // Get model information
    const std::string& getName() const { return modelName_; }
    ModelType getType() const { return modelType_; }
    bool isLoaded() const { return session_ != nullptr; }
    
    // Get input/output shapes
    std::vector<int64_t> getInputShape(size_t index = 0) const;
    std::vector<int64_t> getOutputShape(size_t index = 0) const;
    
protected:
    // Run inference
    std::vector<Ort::Value> runInference(const std::vector<Ort::Value>& inputs);
    
    // Model metadata
    std::string modelName_;
    ModelType modelType_;
    
    // ONNX Runtime objects
    std::unique_ptr<Ort::Session> session_;
    Ort::AllocatorWithDefaultOptions allocator_;
    Ort::MemoryInfo memoryInfo_;
    
    // Input/output names
    std::vector<Ort::AllocatedStringPtr> inputNames_;
    std::vector<Ort::AllocatedStringPtr> outputNames_;
    std::vector<const char*> inputNamePtrs_;
    std::vector<const char*> outputNamePtrs_;
};

// Specialized model wrapper for mel spectrogram computation
class MelSpectrogramModel : public ModelWrapper {
public:
    MelSpectrogramModel();
    
    // Compute mel spectrogram from audio samples
    MelBuffer computeMelSpectrogram(const AudioBuffer& samples);
    
private:
    size_t frameSize_ = 4 * CHUNK_SAMPLES;
};

// Specialized model wrapper for speech embedding
class EmbeddingModel : public ModelWrapper {
public:
    EmbeddingModel();
    
    // Extract embeddings from mel spectrograms
    FeatureBuffer extractEmbeddings(const MelBuffer& mels);
};

// Specialized model wrapper for wake word detection
class WakeWordModel : public ModelWrapper {
public:
    WakeWordModel(const std::string& wakeWord);
    
    // Predict wake word probability from features
    float predict(const FeatureBuffer& features);
    
    // Get wake word this model detects
    const std::string& getWakeWord() const { return wakeWord_; }
    
private:
    std::string wakeWord_;
};

// VAD model wrapper
class VADModel : public ModelWrapper {
public:
    VADModel();
    
    // Predict voice activity probability
    float predictVoiceActivity(const AudioBuffer& samples);
    
    // Reset internal state (for RNN-based models)
    void resetState();
    
private:
    std::vector<float> internalState_;
};

} // namespace openwakeword

#endif // OPENWAKEWORD_MODEL_WRAPPER_H