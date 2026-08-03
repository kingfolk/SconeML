#include "src/tensor/MetalRuntime.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <stdexcept>

namespace sconeml::tensor {

bool metalIsAvailable() {
  @autoreleasepool {
    return MTLCreateSystemDefaultDevice() != nil;
  }
}

std::string metalDeviceName() {
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device == nil)
      return "unavailable";
    return std::string([device.name UTF8String]);
  }
}

void runMetalKernel(const std::string &source, const float *input,
                    float *output, std::size_t elementCount) {
  if (elementCount == 0)
    return;

  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device == nil)
      throw std::runtime_error("no Metal GPU is available");

    NSError *error = nil;
    NSString *sourceString =
        [NSString stringWithUTF8String:source.c_str()];
    id<MTLLibrary> library = [device newLibraryWithSource:sourceString
                                                  options:nil
                                                    error:&error];
    if (library == nil)
      throw std::runtime_error(
          "Metal compilation failed: " +
          std::string([error.localizedDescription UTF8String]));

    id<MTLFunction> function = [library newFunctionWithName:@"tensor_map"];
    if (function == nil)
      throw std::runtime_error("Metal kernel tensor_map was not generated");

    id<MTLComputePipelineState> pipeline =
        [device newComputePipelineStateWithFunction:function error:&error];
    if (pipeline == nil)
      throw std::runtime_error(
          "Metal pipeline creation failed: " +
          std::string([error.localizedDescription UTF8String]));

    const std::size_t byteCount = elementCount * sizeof(float);
    id<MTLBuffer> inputBuffer =
        [device newBufferWithBytes:input
                            length:byteCount
                           options:MTLResourceStorageModeShared];
    id<MTLBuffer> outputBuffer =
        [device newBufferWithLength:byteCount
                            options:MTLResourceStorageModeShared];
    if (inputBuffer == nil || outputBuffer == nil)
      throw std::runtime_error("Metal buffer allocation failed");

    id<MTLCommandQueue> queue = [device newCommandQueue];
    id<MTLCommandBuffer> command = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:inputBuffer offset:0 atIndex:0];
    [encoder setBuffer:outputBuffer offset:0 atIndex:1];
    uint32_t count = static_cast<uint32_t>(elementCount);
    [encoder setBytes:&count length:sizeof(count) atIndex:2];

    NSUInteger width =
        std::min<NSUInteger>(pipeline.maxTotalThreadsPerThreadgroup, 256);
    MTLSize grid = MTLSizeMake(elementCount, 1, 1);
    MTLSize group = MTLSizeMake(width, 1, 1);
    [encoder dispatchThreads:grid threadsPerThreadgroup:group];
    [encoder endEncoding];
    [command commit];
    [command waitUntilCompleted];

    if (command.status == MTLCommandBufferStatusError)
      throw std::runtime_error(
          "Metal execution failed: " +
          std::string([command.error.localizedDescription UTF8String]));

    std::memcpy(output, [outputBuffer contents], byteCount);
  }
}

} // namespace sconeml::tensor
