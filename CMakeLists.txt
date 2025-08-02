cmake_minimum_required(VERSION 3.13)

project(openWakeWord 
    VERSION 1.0.0
    DESCRIPTION "C++ implementation of openWakeWord wake word detection"
    LANGUAGES C CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set default build type if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Compiler flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        add_compile_options(-g -O0)
    else()
        add_compile_options(-O2)
    endif()
endif()

# Platform detection for library paths
if(WIN32)
    if(CMAKE_SIZEOF_VOID_P EQUAL 8)
        set(ARCH_DIR "x64")
    else()
        set(ARCH_DIR "x86")
    endif()
elseif(APPLE)
    # macOS uses different naming
    if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "arm64")
        set(ARCH_DIR "arm64")
    else()
        set(ARCH_DIR "x86_64")
    endif()
elseif(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    set(ARCH_DIR "x86_64")
elseif(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "aarch64")
    set(ARCH_DIR "aarch64")
elseif(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "arm64")
    set(ARCH_DIR "arm64")
else()
    set(ARCH_DIR ${CMAKE_HOST_SYSTEM_PROCESSOR})
endif()

# Set library search path
set(ONNXRUNTIME_LIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/lib/${ARCH_DIR}")

# Find required libraries
find_package(Threads REQUIRED)

# Optional: Find Speex for noise suppression
find_package(PkgConfig)
if(PKG_CONFIG_FOUND)
    pkg_check_modules(SPEEXDSP speexdsp)
    if(SPEEXDSP_FOUND)
        message(STATUS "Found Speex DSP library - noise suppression will be available")
        add_definitions(-DHAVE_SPEEX)
    else()
        message(STATUS "Speex DSP library not found - noise suppression will be disabled")
    endif()
endif()

# Check if ONNX Runtime library exists
if(NOT EXISTS "${ONNXRUNTIME_LIB_DIR}")
    message(FATAL_ERROR "ONNX Runtime library directory not found: ${ONNXRUNTIME_LIB_DIR}\n"
                        "Please download ONNX Runtime and extract to lib/${ARCH_DIR}/")
endif()

# Find ONNX Runtime library
find_library(ONNXRUNTIME_LIB 
    NAMES onnxruntime
    PATHS ${ONNXRUNTIME_LIB_DIR}
    NO_DEFAULT_PATH
    REQUIRED)

if(NOT ONNXRUNTIME_LIB)
    message(FATAL_ERROR "ONNX Runtime library not found in ${ONNXRUNTIME_LIB_DIR}")
endif()

# Source files
set(SOURCES
    src/main.cpp
    src/core/model_wrapper.cpp
    src/core/pipeline.cpp
    src/processors/mel_spectrogram.cpp
    src/processors/speech_embedding.cpp
    src/processors/wake_word_detector.cpp
    src/utils/config.cpp
    src/preprocessors/vad.cpp
    src/preprocessors/speex_noise_suppressor.cpp
)

# Main executable
add_executable(openwakeword ${SOURCES})

# Set properties
target_compile_features(openwakeword PRIVATE cxx_std_20)

# Include directories
target_include_directories(openwakeword PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${ONNXRUNTIME_LIB_DIR}/include
)

# Link libraries
target_link_libraries(openwakeword PRIVATE
    Threads::Threads
    ${ONNXRUNTIME_LIB}
)

# Link optional libraries
if(SPEEXDSP_FOUND)
    target_link_libraries(openwakeword PRIVATE ${SPEEXDSP_LIBRARIES})
    target_include_directories(openwakeword PRIVATE ${SPEEXDSP_INCLUDE_DIRS})
endif()

# Set rpath for finding shared libraries at runtime
if(UNIX AND NOT APPLE)
    set_target_properties(openwakeword PROPERTIES
        INSTALL_RPATH "$ORIGIN/../lib"
        BUILD_RPATH "${ONNXRUNTIME_LIB_DIR}"
    )
elseif(APPLE)
    set_target_properties(openwakeword PROPERTIES
        INSTALL_RPATH "@executable_path/../lib"
        BUILD_RPATH "${ONNXRUNTIME_LIB_DIR}"
    )
endif()

# Copy ONNX Runtime shared libraries to output directory
if(WIN32)
    file(GLOB ONNX_RUNTIME_LIBS "${ONNXRUNTIME_LIB_DIR}/*.dll")
else()
    file(GLOB ONNX_RUNTIME_LIBS "${ONNXRUNTIME_LIB_DIR}/*.so*" "${ONNXRUNTIME_LIB_DIR}/*.dylib*")
endif()

foreach(lib ${ONNX_RUNTIME_LIBS})
    add_custom_command(TARGET openwakeword POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${lib}
        $<TARGET_FILE_DIR:openwakeword>
        COMMENT "Copying ONNX Runtime library: ${lib}"
    )
endforeach()

# Installation rules
install(TARGETS openwakeword
    RUNTIME DESTINATION bin
)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/models/
    DESTINATION share/openwakeword/models
    FILES_MATCHING PATTERN "*.onnx"
)

install(FILES ${ONNX_RUNTIME_LIBS}
    DESTINATION lib
)

# Print configuration summary
message(STATUS "")
message(STATUS "openWakeWord Configuration:")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Architecture: ${ARCH_DIR}")
message(STATUS "  ONNX Runtime: ${ONNXRUNTIME_LIB}")
message(STATUS "  Output directory: ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
message(STATUS "")