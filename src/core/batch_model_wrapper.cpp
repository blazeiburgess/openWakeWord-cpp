#include "core/batch_model_wrapper.h"
#include <iostream>
#include <numeric>

namespace openwakeword {

// BatchMelSpectrogramModel implementation
BatchMelSpectrogramModel::BatchMelSpectrogramModel() 
    : BatchModelWrapper("MelSpectrogram", ModelType::MELSPECTROGRAM, 16) {
}

MelBuffer BatchMelSpectrogramModel::inference(const AudioBuffer& samples) {
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
    const float* melData = melOut.GetTensorData<float>();
    const auto melInfo = melOut.GetTensorTypeAndShapeInfo();
    const auto melShape = melInfo.GetShape();
    
    size_t melCount = std::accumulate(melShape.begin(), melShape.end(), 1, std::multiplies<>());
    
    return MelBuffer(melData, melData + melCount);
}

std::vector<MelBuffer> BatchMelSpectrogramModel::batchInference(const std::vector<AudioBuffer>& sampleBatches) {
    if (sampleBatches.empty()) {
        return {};
    }
    
    size_t batchSize = sampleBatches.size();
    
    // Validate all inputs have correct size
    for (const auto& samples : sampleBatches) {
        if (samples.size() != frameSize_) {
            throw std::invalid_argument("Invalid sample buffer size in batch");
        }
    }
    
    // Create batched input tensor
    std::vector<float> batchedData;
    batchedData.reserve(batchSize * frameSize_);
    for (const auto& samples : sampleBatches) {
        batchedData.insert(batchedData.end(), samples.begin(), samples.end());
    }
    
    std::vector<int64_t> inputShape{static_cast<int64_t>(batchSize), static_cast<int64_t>(frameSize_)};
    std::vector<Ort::Value> inputs;
    inputs.push_back(Ort::Value::CreateTensor<float>(
        memoryInfo_, batchedData.data(), batchedData.size(),
        inputShape.data(), inputShape.size()));
    
    // Run inference
    auto outputs = runInference(inputs);
    
    // Extract batched mel spectrograms
    const auto& melOut = outputs.front();
    const float* melData = melOut.GetTensorData<float>();
    const auto melInfo = melOut.GetTensorTypeAndShapeInfo();
    const auto melShape = melInfo.GetShape();
    
    // Assume shape is [batch_size, time, mel_bins]
    size_t melPerSample = (melShape.size() > 1) ? 
        std::accumulate(melShape.begin() + 1, melShape.end(), 1, std::multiplies<>()) : 1;
    
    std::vector<MelBuffer> results;
    results.reserve(batchSize);
    
    for (size_t i = 0; i < batchSize; ++i) {
        const float* sampleStart = melData + (i * melPerSample);
        results.emplace_back(sampleStart, sampleStart + melPerSample);
    }
    
    return results;
}

// BatchEmbeddingModel implementation
BatchEmbeddingModel::BatchEmbeddingModel() 
    : BatchModelWrapper("SpeechEmbedding", ModelType::EMBEDDING, 8) {
}

FeatureBuffer BatchEmbeddingModel::inference(const MelBuffer& mels) {
    // Create input tensor
    std::vector<int64_t> inputShape{1, static_cast<int64_t>(mels.size())};
    std::vector<Ort::Value> inputs;
    inputs.push_back(Ort::Value::CreateTensor<float>(
        memoryInfo_, const_cast<float*>(mels.data()), mels.size(),
        inputShape.data(), inputShape.size()));
    
    // Run inference
    auto outputs = runInference(inputs);
    
    // Extract embeddings
    const auto& embOut = outputs.front();
    const float* embData = embOut.GetTensorData<float>();
    const auto embInfo = embOut.GetTensorTypeAndShapeInfo();
    const auto embShape = embInfo.GetShape();
    
    size_t embCount = std::accumulate(embShape.begin(), embShape.end(), 1, std::multiplies<>());
    
    return FeatureBuffer(embData, embData + embCount);
}

std::vector<FeatureBuffer> BatchEmbeddingModel::batchInference(const std::vector<MelBuffer>& melBatches) {
    if (melBatches.empty()) {
        return {};
    }
    
    size_t batchSize = melBatches.size();
    size_t melSize = melBatches[0].size();
    
    // Create batched input tensor
    std::vector<float> batchedData;
    batchedData.reserve(batchSize * melSize);
    for (const auto& mels : melBatches) {
        if (mels.size() != melSize) {
            throw std::invalid_argument("Inconsistent mel buffer sizes in batch");
        }
        batchedData.insert(batchedData.end(), mels.begin(), mels.end());
    }
    
    std::vector<int64_t> inputShape{static_cast<int64_t>(batchSize), static_cast<int64_t>(melSize)};
    std::vector<Ort::Value> inputs;
    inputs.push_back(Ort::Value::CreateTensor<float>(
        memoryInfo_, batchedData.data(), batchedData.size(),
        inputShape.data(), inputShape.size()));
    
    // Run inference
    auto outputs = runInference(inputs);
    
    // Extract batched embeddings
    const auto& embOut = outputs.front();
    const float* embData = embOut.GetTensorData<float>();
    const auto embInfo = embOut.GetTensorTypeAndShapeInfo();
    const auto embShape = embInfo.GetShape();
    
    size_t embPerSample = (embShape.size() > 1) ? 
        std::accumulate(embShape.begin() + 1, embShape.end(), 1, std::multiplies<>()) : 1;
    
    std::vector<FeatureBuffer> results;
    results.reserve(batchSize);
    
    for (size_t i = 0; i < batchSize; ++i) {
        const float* sampleStart = embData + (i * embPerSample);
        results.emplace_back(sampleStart, sampleStart + embPerSample);
    }
    
    return results;
}

// BatchWakeWordModel implementation
BatchWakeWordModel::BatchWakeWordModel(const std::string& wakeWord)
    : BatchModelWrapper("WakeWord_" + wakeWord, ModelType::WAKEWORD, 32),
      wakeWord_(wakeWord) {
}

float BatchWakeWordModel::inference(const FeatureBuffer& features) {
    // Create input tensor
    std::vector<int64_t> inputShape{1, static_cast<int64_t>(features.size())};
    std::vector<Ort::Value> inputs;
    inputs.push_back(Ort::Value::CreateTensor<float>(
        memoryInfo_, const_cast<float*>(features.data()), features.size(),
        inputShape.data(), inputShape.size()));
    
    // Run inference
    auto outputs = runInference(inputs);
    
    // Extract probability
    const auto& probOut = outputs.front();
    const float* probData = probOut.GetTensorData<float>();
    
    return probData[0];
}

std::vector<float> BatchWakeWordModel::batchInference(const std::vector<FeatureBuffer>& featureBatches) {
    if (featureBatches.empty()) {
        return {};
    }
    
    size_t batchSize = featureBatches.size();
    size_t featureSize = featureBatches[0].size();
    
    // Create batched input tensor
    std::vector<float> batchedData;
    batchedData.reserve(batchSize * featureSize);
    for (const auto& features : featureBatches) {
        if (features.size() != featureSize) {
            throw std::invalid_argument("Inconsistent feature buffer sizes in batch");
        }
        batchedData.insert(batchedData.end(), features.begin(), features.end());
    }
    
    std::vector<int64_t> inputShape{static_cast<int64_t>(batchSize), static_cast<int64_t>(featureSize)};
    std::vector<Ort::Value> inputs;
    inputs.push_back(Ort::Value::CreateTensor<float>(
        memoryInfo_, batchedData.data(), batchedData.size(),
        inputShape.data(), inputShape.size()));
    
    // Run inference
    auto outputs = runInference(inputs);
    
    // Extract probabilities
    const auto& probOut = outputs.front();
    const float* probData = probOut.GetTensorData<float>();
    const auto probInfo = probOut.GetTensorTypeAndShapeInfo();
    const auto probShape = probInfo.GetShape();
    
    // Assume output shape is [batch_size] or [batch_size, 1]
    std::vector<float> results;
    results.reserve(batchSize);
    
    for (size_t i = 0; i < batchSize; ++i) {
        results.push_back(probData[i]);
    }
    
    return results;
}

} // namespace openwakeword