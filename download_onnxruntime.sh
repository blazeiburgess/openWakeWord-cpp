#!/bin/bash

# Script to download ONNX Runtime library

ARCH=$(uname -m)
ONNX_VERSION="1.21.1"

echo "Detected architecture: $ARCH"

# Determine the correct ONNX Runtime package
case "$ARCH" in
    "x86_64")
        ONNX_ARCH="x64"
        ONNX_FILE="onnxruntime-linux-x64-gpu-${ONNX_VERSION}"
        ;;
    "aarch64")
        ONNX_ARCH="aarch64"
        ONNX_FILE="onnxruntime-linux-aarch64-${ONNX_VERSION}"
        ;;
    *)
        echo "Unsupported architecture: $ARCH"
        exit 1
        ;;
esac

DOWNLOAD_URL="https://github.com/microsoft/onnxruntime/releases/download/v${ONNX_VERSION}/${ONNX_FILE}.tgz"
DEST_DIR="lib/${ARCH}"

echo "Creating directory: $DEST_DIR"
mkdir -p "$DEST_DIR"

echo "Downloading ONNX Runtime ${ONNX_VERSION} for ${ARCH}..."
echo "URL: $DOWNLOAD_URL"

# Download and extract
cd "$DEST_DIR" || exit 1
curl -L -o onnxruntime.tgz "$DOWNLOAD_URL"

echo "Extracting..."
tar -xzf onnxruntime.tgz

# Move files to the correct location
mv ${ONNX_FILE}/lib/* .
if [ -d "${ONNX_FILE}/include" ]; then
    cp -r ${ONNX_FILE}/include .
fi

# Clean up
rm -rf ${ONNX_FILE} onnxruntime.tgz

echo "ONNX Runtime installed successfully in $DEST_DIR"
echo ""
echo "You can now build the project with:"
echo "  mkdir build && cd build"
echo "  cmake .."
echo "  make"