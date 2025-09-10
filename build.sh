#!/bin/bash

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Building MLIR Minimal Dialect...${NC}"

# We pass LLVM_DIR manually, so comment following check step
# # Check path for llvm and mlir
# if ! command -v llvm-config &> /dev/null; then
#     echo -e "${RED}Error: llvm-config not found${NC}"
#     exit 1
# fi

BUILD_DIR="build"
if [ -d "$BUILD_DIR" ]; then
    echo -e "${YELLOW}Cleanup...${NC}"
    rm -rf "$BUILD_DIR"
fi

mkdir "$BUILD_DIR"
cd "$BUILD_DIR"

echo -e "${GREEN}Config CMake...${NC}"
cmake .. \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON\
    -DLLVM_DIR=/usr/lib/llvm-20/lib/cmake/llvm

if [ $? -ne 0 ]; then
    echo -e "${RED}CMake config failed${NC}"
    exit 1
fi

echo -e "${GREEN}Making...${NC}"
make -j$(nproc)

if [ $? -ne 0 ]; then
    echo -e "${RED}Making failed${NC}"
    exit 1
fi

echo -e "${GREEN}Build success!${NC}"

# echo -e "${GREEN}Exec Path: ${BUILD_DIR}/mlir_example${NC}"

# echo -e "${GREEN}Run example...${NC}"
# ./mlir_example

# echo -e "${GREEN}Complete!${NC}"