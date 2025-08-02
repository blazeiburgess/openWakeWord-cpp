# PowerShell script to download ONNX Runtime library for Windows

param(
    [string]$OnnxVersion = "1.21.1",
    [string]$Architecture = "x64"
)

Write-Host "OpenWakeWord ONNX Runtime Download Script for Windows" -ForegroundColor Cyan
Write-Host "====================================================" -ForegroundColor Cyan
Write-Host ""

# Determine architecture if not specified
if ($Architecture -eq "") {
    if ([Environment]::Is64BitOperatingSystem) {
        $Architecture = "x64"
    } else {
        $Architecture = "x86"
    }
}

Write-Host "ONNX Runtime Version: $OnnxVersion" -ForegroundColor Green
Write-Host "Architecture: $Architecture" -ForegroundColor Green

# Set download parameters
$onnxFile = "onnxruntime-win-$Architecture-gpu-$OnnxVersion"
$downloadUrl = "https://github.com/microsoft/onnxruntime/releases/download/v$OnnxVersion/$onnxFile.zip"
$destDir = "lib\$Architecture"

# Create destination directory
Write-Host "`nCreating directory: $destDir" -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path $destDir | Out-Null

# Download ONNX Runtime
Write-Host "Downloading ONNX Runtime from:" -ForegroundColor Yellow
Write-Host "$downloadUrl" -ForegroundColor Gray
Write-Host ""

try {
    $progressPreference = 'silentlyContinue'
    Invoke-WebRequest -Uri $downloadUrl -OutFile "$destDir\onnxruntime.zip" -UseBasicParsing
    $progressPreference = 'Continue'
    Write-Host "Download completed successfully!" -ForegroundColor Green
} catch {
    Write-Host "ERROR: Failed to download ONNX Runtime" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    exit 1
}

# Extract the archive
Write-Host "`nExtracting archive..." -ForegroundColor Yellow
try {
    Expand-Archive -Path "$destDir\onnxruntime.zip" -DestinationPath $destDir -Force
    Write-Host "Extraction completed!" -ForegroundColor Green
} catch {
    Write-Host "ERROR: Failed to extract archive" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    exit 1
}

# Move files to the correct location
Write-Host "`nOrganizing files..." -ForegroundColor Yellow

# Move all files from lib subdirectory
$libPath = Join-Path $destDir "$onnxFile\lib\*"
if (Test-Path $libPath) {
    Get-ChildItem $libPath | Move-Item -Destination $destDir -Force
    Write-Host "  - Moved library files" -ForegroundColor Gray
}

# Copy include directory if it exists
$includePath = Join-Path $destDir "$onnxFile\include"
if (Test-Path $includePath) {
    Copy-Item -Path $includePath -Destination $destDir -Recurse -Force
    Write-Host "  - Copied include files" -ForegroundColor Gray
}

# Clean up
Write-Host "`nCleaning up temporary files..." -ForegroundColor Yellow
Remove-Item -Path "$destDir\$onnxFile" -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item -Path "$destDir\onnxruntime.zip" -Force

Write-Host "`nONNX Runtime installed successfully!" -ForegroundColor Green
Write-Host "Location: $destDir" -ForegroundColor Gray
Write-Host ""
Write-Host "You can now build the project with:" -ForegroundColor Cyan
Write-Host "  mkdir build" -ForegroundColor White
Write-Host "  cd build" -ForegroundColor White
Write-Host '  cmake .. -G "Visual Studio 17 2022"' -ForegroundColor White
Write-Host "  cmake --build . --config Release" -ForegroundColor White
Write-Host ""

# List installed files
Write-Host "Installed files:" -ForegroundColor Yellow
Get-ChildItem $destDir -Name | ForEach-Object { Write-Host "  - $_" -ForegroundColor Gray }