#!/bin/bash

# Script to download ONNX Runtime library for Linux and macOS

ARCH=$(uname -m)
OS=$(uname -s)
ONNX_VERSION="1.21.1"

echo "OpenWakeWord ONNX Runtime Download Script"
echo "========================================="
echo "Detected OS: $OS"
echo "Detected architecture: $ARCH"
echo ""

# Determine the correct ONNX Runtime package based on OS and architecture
if [ "$OS" = "Darwin" ]; then
    # macOS
    case "$ARCH" in
        "x86_64")
            ONNX_FILE="onnxruntime-osx-x86_64-${ONNX_VERSION}"
            ;;
        "arm64")
            ONNX_FILE="onnxruntime-osx-arm64-${ONNX_VERSION}"
            ;;
        *)
            echo "Unsupported macOS architecture: $ARCH"
            exit 1
            ;;
    esac
elif [ "$OS" = "Linux" ]; then
    # Linux
    case "$ARCH" in
        "x86_64")
            ONNX_FILE="onnxruntime-linux-x64-gpu-${ONNX_VERSION}"
            ;;
        "aarch64")
            ONNX_FILE="onnxruntime-linux-aarch64-${ONNX_VERSION}"
            ;;
        *)
            echo "Unsupported Linux architecture: $ARCH"
            exit 1
            ;;
    esac
else
    echo "Unsupported operating system: $OS"
    echo "This script supports Linux and macOS only."
    echo "For Windows, please use download_onnxruntime.ps1"
    exit 1
fi

DOWNLOAD_URL="https://github.com/microsoft/onnxruntime/releases/download/v${ONNX_VERSION}/${ONNX_FILE}.tgz"
DEST_DIR="lib/${ARCH}"

echo "ONNX Runtime package: $ONNX_FILE"
echo "Destination directory: $DEST_DIR"
echo ""

# Create destination directory
echo "Creating directory: $DEST_DIR"
mkdir -p "$DEST_DIR"

# Download ONNX Runtime
echo "Downloading ONNX Runtime ${ONNX_VERSION}..."
echo "URL: $DOWNLOAD_URL"
echo ""

cd "$DEST_DIR" || exit 1

if command -v curl &> /dev/null; then
    curl -L -o onnxruntime.tgz "$DOWNLOAD_URL"
elif command -v wget &> /dev/null; then
    wget -O onnxruntime.tgz "$DOWNLOAD_URL"
else
    echo "ERROR: Neither curl nor wget is available. Please install one of them."
    exit 1
fi

if [ ! -f "onnxruntime.tgz" ]; then
    echo "ERROR: Failed to download ONNX Runtime"
    exit 1
fi

# Extract the archive
echo ""
echo "Extracting archive..."
tar -xzf onnxruntime.tgz

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to extract archive"
    exit 1
fi

# Move files to the correct location
echo "Organizing files..."
if [ -d "${ONNX_FILE}/lib" ]; then
    mv ${ONNX_FILE}/lib/* .
    echo "  - Moved library files"
fi

if [ -d "${ONNX_FILE}/include" ]; then
    cp -r ${ONNX_FILE}/include .
    echo "  - Copied include files"
fi

# Clean up
echo "Cleaning up temporary files..."
rm -rf ${ONNX_FILE} onnxruntime.tgz

echo ""
echo "ONNX Runtime installed successfully in $DEST_DIR"
echo ""

# Platform-specific build instructions
if [ "$OS" = "Darwin" ]; then
    echo "You can now build the project with:"
    echo "  mkdir build && cd build"
    echo "  cmake .."
    echo "  make -j\$(sysctl -n hw.ncpu)"
    echo ""
    echo "For universal binary (Intel + Apple Silicon):"
    echo '  cmake .. -DCMAKE_OSX_ARCHITECTURES="x86_64;arm64"'
else
    echo "You can now build the project with:"
    echo "  mkdir build && cd build"
    echo "  cmake .."
    echo "  make -j\$(nproc)"
fi

# List installed files
echo ""
echo "Installed files:"
ls -la | grep -v "^d" | awk '{print "  - " $9}' | grep -v "^  - $"