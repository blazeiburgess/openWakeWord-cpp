#ifndef SIMD_AUDIO_H
#define SIMD_AUDIO_H

#include <vector>
#include <cstdint>
#include "core/types.h"

// Platform detection
#if defined(__SSE2__)
#include <emmintrin.h>
#endif

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace openwakeword {

class SimdAudio {
public:
    // Convert int16_t to float with SIMD acceleration
    static void convertToFloat(const int16_t* input, float* output, size_t count);
    
    // Convert int16_t to float into vector
    static void convertToFloat(const int16_t* input, std::vector<float>& output, size_t count);
    
    // Check if SIMD is available
    static bool isSimdAvailable();
    
private:
#if defined(__SSE2__)
    static void convertToFloatSSE2(const int16_t* input, float* output, size_t count);
#endif
    
#if defined(__ARM_NEON)
    static void convertToFloatNEON(const int16_t* input, float* output, size_t count);
#endif
    
    // Fallback scalar implementation
    static void convertToFloatScalar(const int16_t* input, float* output, size_t count);
};

// Inline implementations for optimal performance

inline void SimdAudio::convertToFloat(const int16_t* input, float* output, size_t count) {
#if defined(__SSE2__)
    convertToFloatSSE2(input, output, count);
#elif defined(__ARM_NEON)
    convertToFloatNEON(input, output, count);
#else
    convertToFloatScalar(input, output, count);
#endif
}

inline void SimdAudio::convertToFloat(const int16_t* input, std::vector<float>& output, size_t count) {
    output.resize(count);
    convertToFloat(input, output.data(), count);
}

inline bool SimdAudio::isSimdAvailable() {
#if defined(__SSE2__) || defined(__ARM_NEON)
    return true;
#else
    return false;
#endif
}

#if defined(__SSE2__)
inline void SimdAudio::convertToFloatSSE2(const int16_t* input, float* output, size_t count) {
    size_t simd_count = count & ~7;  // Process 8 at a time
    size_t i = 0;
    
    // Process 8 samples at a time
    for (; i < simd_count; i += 8) {
        // Load 8 int16_t values
        __m128i v_int16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(input + i));
        
        // Split into two groups of 4 and convert to int32
        __m128i v_int32_lo = _mm_srai_epi32(_mm_unpacklo_epi16(v_int16, v_int16), 16);
        __m128i v_int32_hi = _mm_srai_epi32(_mm_unpackhi_epi16(v_int16, v_int16), 16);
        
        // Convert to float
        __m128 v_float_lo = _mm_cvtepi32_ps(v_int32_lo);
        __m128 v_float_hi = _mm_cvtepi32_ps(v_int32_hi);
        
        // Store results
        _mm_storeu_ps(output + i, v_float_lo);
        _mm_storeu_ps(output + i + 4, v_float_hi);
    }
    
    // Handle remaining samples
    for (; i < count; ++i) {
        output[i] = static_cast<float>(input[i]);
    }
}
#endif

#if defined(__ARM_NEON)
inline void SimdAudio::convertToFloatNEON(const int16_t* input, float* output, size_t count) {
    size_t simd_count = count & ~7;  // Process 8 at a time
    size_t i = 0;
    
    // Process 8 samples at a time
    for (; i < simd_count; i += 8) {
        // Load 8 int16_t values
        int16x8_t v_int16 = vld1q_s16(input + i);
        
        // Split into two groups of 4 and convert to int32
        int32x4_t v_int32_lo = vmovl_s16(vget_low_s16(v_int16));
        int32x4_t v_int32_hi = vmovl_s16(vget_high_s16(v_int16));
        
        // Convert to float
        float32x4_t v_float_lo = vcvtq_f32_s32(v_int32_lo);
        float32x4_t v_float_hi = vcvtq_f32_s32(v_int32_hi);
        
        // Store results
        vst1q_f32(output + i, v_float_lo);
        vst1q_f32(output + i + 4, v_float_hi);
    }
    
    // Handle remaining samples
    for (; i < count; ++i) {
        output[i] = static_cast<float>(input[i]);
    }
}
#endif

inline void SimdAudio::convertToFloatScalar(const int16_t* input, float* output, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        output[i] = static_cast<float>(input[i]);
    }
}

} // namespace openwakeword

#endif // SIMD_AUDIO_H