#include "core/model_wrapper.h"
#include <iostream>
#include <numeric>

namespace openwakeword {

ModelWrapper::ModelWrapper(const std::string& modelName, ModelType type)
    : modelName_(modelName), modelType_(type),
      memoryInfo_(Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, 
                                             OrtMemType::OrtMemTypeDefault)) {
}

bool ModelWrapper::loadModel(const std::filesystem::path& modelPath, 
                            Ort::Env& env,
                            const Ort::SessionOptions& options) {
    try {
        session_ = std::make_unique<Ort::Session>(env, modelPath.c_str(), options);
        
        // Get input names
        size_t numInputs = session_->GetInputCount();
        inputNames_.clear();
        inputNamePtrs_.clear();
        for (size_t i = 0; i < numInputs; ++i) {
            inputNames_.push_back(session_->GetInputNameAllocated(i, allocator_));
            inputNamePtrs_.push_back(inputNames_.back().get());
        }
        
        // Get output names
        size_t numOutputs = session_->GetOutputCount();
        outputNames_.clear();
        outputNamePtrs_.clear();
        for (size_t i = 0; i < numOutputs; ++i) {
            outputNames_.push_back(session_->GetOutputNameAllocated(i, allocator_));
            outputNamePtrs_.push_back(outputNames_.back().get());
        }
        
        return true;
    } catch (const Ort::Exception& e) {
        std::cerr << "[ERROR] Failed to load model " << modelPath << ": " << e.what() << std::endl;
        return false;
    }
}

std::vector<int64_t> ModelWrapper::getInputShape(size_t index) const {
    if (!session_ || index >= session_->GetInputCount()) {
        return {};
    }
    
    auto typeInfo = session_->GetInputTypeInfo(index);
    auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();
    return tensorInfo.GetShape();
}

std::vector<int64_t> ModelWrapper::getOutputShape(size_t index) const {
    if (!session_ || index >= session_->GetOutputCount()) {
        return {};
    }
    
    auto typeInfo = session_->GetOutputTypeInfo(index);
    auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();
    return tensorInfo.GetShape();
}

std::vector<Ort::Value> ModelWrapper::runInference(const std::vector<Ort::Value>& inputs) {
    if (!session_) {
        throw std::runtime_error("Model not loaded");
    }
    
    return session_->Run(Ort::RunOptions{nullptr}, 
                        inputNamePtrs_.data(), inputs.data(), inputs.size(),
                        outputNamePtrs_.data(), outputNamePtrs_.size());
}

// MelSpectrogramModel implementation
MelSpectrogramModel::MelSpectrogramModel() 
    : ModelWrapper("MelSpectrogram", ModelType::MELSPECTROGRAM) {
}

MelBuffer MelSpectrogramModel::computeMelSpectrogram(const AudioBuffer& samples) {
    if (samples.size() != frameSize_) {
        throw std::invalid_argument("Invalid sample buffer size");
    }
    
    // Create input tensor
    std::vector<int64_t> inputShape{1, static_cast<int64_t>(frameSize_)};
    std::vector<Ort::Value> inputs;
    inputs.push_back(Ort::Value::CreateTensor<float>(
        memoryInfo_, const_cast<float*>(samples.data()), frameSize_,
        inputShape.data(), inputShape.size()));
    
    // Run inference
    auto outputs = runInference(inputs);
    
    // Extract mel spectrogram data
    const auto& melOut = outputs.front();
    const auto melInfo = melOut.GetTensorTypeAndShapeInfo();
    const auto melShape = melInfo.GetShape();
    
    const float* melData = melOut.GetTensorData<float>();
    size_t melCount = std::accumulate(melShape.begin(), melShape.end(), 1, std::multiplies<>());
    
    // Scale mels for Google speech embedding model
    MelBuffer result;
    result.reserve(melCount);
    for (size_t i = 0; i < melCount; ++i) {
        result.push_back((melData[i] / 10.0f) + 2.0f);
    }
    
    return result;
}

// EmbeddingModel implementation
EmbeddingModel::EmbeddingModel()
    : ModelWrapper("SpeechEmbedding", ModelType::EMBEDDING) {
}

FeatureBuffer EmbeddingModel::extractEmbeddings(const MelBuffer& mels) {
    size_t expectedSize = EMBEDDING_WINDOW_SIZE * NUM_MELS;
    if (mels.size() < expectedSize) {
        throw std::invalid_argument("Insufficient mel data for embedding extraction");
    }
    
    // Create input tensor
    std::vector<int64_t> inputShape{1, static_cast<int64_t>(EMBEDDING_WINDOW_SIZE), 
                                   static_cast<int64_t>(NUM_MELS), 1};
    std::vector<Ort::Value> inputs;
    inputs.push_back(Ort::Value::CreateTensor<float>(
        memoryInfo_, const_cast<float*>(mels.data()), expectedSize,
        inputShape.data(), inputShape.size()));
    
    // Run inference
    auto outputs = runInference(inputs);
    
    // Extract embedding data
    const auto& embOut = outputs.front();
    const float* embData = embOut.GetTensorData<float>();
    
    const auto embInfo = embOut.GetTensorTypeAndShapeInfo();
    const auto embShape = embInfo.GetShape();
    size_t embCount = std::accumulate(embShape.begin(), embShape.end(), 1, std::multiplies<>());
    
    return FeatureBuffer(embData, embData + embCount);
}

// WakeWordModel implementation
WakeWordModel::WakeWordModel(const std::string& wakeWord)
    : ModelWrapper(wakeWord, ModelType::WAKEWORD), wakeWord_(wakeWord) {
}

float WakeWordModel::predict(const FeatureBuffer& features) {
    size_t expectedSize = WAKEWORD_FEATURES * EMBEDDING_FEATURES;
    if (features.size() < expectedSize) {
        throw std::invalid_argument("Insufficient features for wake word detection");
    }
    
    // Create input tensor
    std::vector<int64_t> inputShape{1, static_cast<int64_t>(WAKEWORD_FEATURES), 
                                   static_cast<int64_t>(EMBEDDING_FEATURES)};
    std::vector<Ort::Value> inputs;
    inputs.push_back(Ort::Value::CreateTensor<float>(
        memoryInfo_, const_cast<float*>(features.data()), expectedSize,
        inputShape.data(), inputShape.size()));
    
    // Run inference
    auto outputs = runInference(inputs);
    
    // Extract probability
    const auto& out = outputs.front();
    const float* data = out.GetTensorData<float>();
    
    return data[0];  // Assuming single output
}

// VADModel implementation
VADModel::VADModel()
    : ModelWrapper("VAD", ModelType::VAD) {
}

float VADModel::predictVoiceActivity(const AudioBuffer& samples) {
    // TODO: Implement VAD prediction
    // This will depend on the specific VAD model architecture
    (void)samples; // Suppress unused parameter warning
    return 1.0f;  // Placeholder
}

void VADModel::resetState() {
    internalState_.clear();
}

} // namespace openwakeword