#!/bin/bash

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Building MLIR Minimal Dialect...${NC}"

BREW_BIN="$(command -v brew || true)"
if [ -z "$BREW_BIN" ] && [ -x /opt/homebrew/bin/brew ]; then
    BREW_BIN=/opt/homebrew/bin/brew
fi

if [ -z "$BREW_BIN" ]; then
    echo -e "${RED}Error: Homebrew is required (https://brew.sh)${NC}"
    exit 1
fi

LLVM_PREFIX="$($BREW_BIN --prefix llvm)"
CMAKE_BIN="$($BREW_BIN --prefix cmake)/bin/cmake"

if [ ! -x "$CMAKE_BIN" ]; then
    echo -e "${RED}Error: Homebrew CMake is required (brew install cmake)${NC}"
    exit 1
fi

BUILD_DIR="build"
if [ ! -d "$BUILD_DIR" ]; then
    echo -e "${YELLOW}Creating build directory...${NC}"
    mkdir "$BUILD_DIR"
fi

cd "$BUILD_DIR"

echo -e "${GREEN}Config CMake...${NC}"
"$CMAKE_BIN" .. \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DCMAKE_C_COMPILER="$LLVM_PREFIX/bin/clang" \
    -DCMAKE_CXX_COMPILER="$LLVM_PREFIX/bin/clang++" \
    -DLLVM_DIR="$LLVM_PREFIX/lib/cmake/llvm" \
    -DMLIR_DIR="$LLVM_PREFIX/lib/cmake/mlir" \
    -DCMAKE_PREFIX_PATH="$($BREW_BIN --prefix zstd)"

if [ $? -ne 0 ]; then
    echo -e "${RED}CMake config failed${NC}"
    exit 1
fi

echo -e "${GREEN}Making...${NC}"
make -j"$(sysctl -n hw.ncpu)"

if [ $? -ne 0 ]; then
    echo -e "${RED}Making failed${NC}"
    exit 1
fi

echo -e "${GREEN}Build success!${NC}"

# echo -e "${GREEN}Exec Path: ${BUILD_DIR}/mlir_example${NC}"

# echo -e "${GREEN}Run example...${NC}"
# ./mlir_example

# echo -e "${GREEN}Complete!${NC}"
