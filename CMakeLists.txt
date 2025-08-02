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
    elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
        # Aggressive optimizations for Release builds
        add_compile_options(-O3)
        
        # CPU-specific optimizations if not cross-compiling
        if(NOT CMAKE_CROSSCOMPILING AND NOT DEFINED CMAKE_OSX_ARCHITECTURES)
            include(CheckCXXCompilerFlag)
            CHECK_CXX_COMPILER_FLAG("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
            if(COMPILER_SUPPORTS_MARCH_NATIVE)
                add_compile_options(-march=native)
            endif()
        endif()
        
        # Link Time Optimization
        include(CheckIPOSupported)
        check_ipo_supported(RESULT IPO_SUPPORTED)
        if(IPO_SUPPORTED)
            set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
        endif()
    else()
        add_compile_options(-O2)
    endif()
endif()

# Profile-Guided Optimization support
option(ENABLE_PGO_GENERATE "Enable PGO profile generation" OFF)
option(ENABLE_PGO_USE "Enable PGO profile use" OFF)

if(ENABLE_PGO_GENERATE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-fprofile-generate)
    add_link_options(-fprofile-generate)
    message(STATUS "PGO profile generation enabled")
elseif(ENABLE_PGO_USE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
        add_compile_options(-fprofile-use)
        add_link_options(-fprofile-use)
    else() # Clang
        if(EXISTS "${CMAKE_BINARY_DIR}/default.profdata")
            add_compile_options(-fprofile-use=${CMAKE_BINARY_DIR}/default.profdata)
            add_link_options(-fprofile-use=${CMAKE_BINARY_DIR}/default.profdata)
        else()
            message(WARNING "PGO profile data not found at ${CMAKE_BINARY_DIR}/default.profdata")
        endif()
    endif()
    message(STATUS "PGO profile use enabled")
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
    pkg_check_modules(SPEEXDSP IMPORTED_TARGET speexdsp)
    if(SPEEXDSP_FOUND)
        message(STATUS "Found Speex DSP library - noise suppression will be available")
        add_definitions(-DHAVE_SPEEX)
        
        # Debug output for pkg-config variables
        message(STATUS "SPEEXDSP_VERSION: ${SPEEXDSP_VERSION}")
        message(STATUS "SPEEXDSP_PREFIX: ${SPEEXDSP_PREFIX}")
        message(STATUS "SPEEXDSP_LIBDIR: ${SPEEXDSP_LIBDIR}")
        message(STATUS "SPEEXDSP_LIBRARY_DIRS: ${SPEEXDSP_LIBRARY_DIRS}")
        message(STATUS "SPEEXDSP_LIBRARIES: ${SPEEXDSP_LIBRARIES}")
        message(STATUS "SPEEXDSP_LINK_LIBRARIES: ${SPEEXDSP_LINK_LIBRARIES}")
        message(STATUS "SPEEXDSP_INCLUDE_DIRS: ${SPEEXDSP_INCLUDE_DIRS}")
        message(STATUS "SPEEXDSP_CFLAGS: ${SPEEXDSP_CFLAGS}")
        message(STATUS "SPEEXDSP_LDFLAGS: ${SPEEXDSP_LDFLAGS}")
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
    src/core/batch_model_wrapper.cpp
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
    target_link_libraries(openwakeword PRIVATE PkgConfig::SPEEXDSP)
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
elseif(APPLE)
    file(GLOB ONNX_RUNTIME_LIBS "${ONNXRUNTIME_LIB_DIR}/*.dylib")
else()
    file(GLOB ONNX_RUNTIME_LIBS "${ONNXRUNTIME_LIB_DIR}/*.so*")
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
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    message(STATUS "  Optimizations:")
    if(COMPILER_SUPPORTS_MARCH_NATIVE)
        message(STATUS "    - CPU-specific optimizations: enabled")
    endif()
    if(IPO_SUPPORTED)
        message(STATUS "    - Link Time Optimization: enabled")
    endif()
    if(ENABLE_PGO_GENERATE)
        message(STATUS "    - PGO profile generation: enabled")
    elseif(ENABLE_PGO_USE)
        message(STATUS "    - PGO profile use: enabled")
    endif()
endif()
message(STATUS "")