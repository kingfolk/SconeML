#include "src/tensor/MetalRuntime.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
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

void runMetalKernel(const std::string &source, const std::string &entrypoint,
                    std::size_t exportedInputCount,
                    std::span<const MetalInput> inputs, std::span<int32_t> output,
                    std::size_t elementCount, MetalLaunchOptions options) {
  // Reject host-side errors before touching Metal, including in CPU-only runs.
  validateMetalInputs(exportedInputCount, inputs, elementCount);
  planMetalLaunch(elementCount, elementCount, output.size(), options);
  if (elementCount == 0)
    return;
  if (!output.data())
    throw std::runtime_error("Metal nonempty buffers require nonnull pointers");
  if (entrypoint.empty())
    throw std::runtime_error("Metal entrypoint must not be empty");

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

    NSString *functionName = [NSString stringWithUTF8String:entrypoint.c_str()];
    id<MTLFunction> function = [library newFunctionWithName:functionName];
    if (function == nil)
      throw std::runtime_error("Metal kernel entrypoint was not generated: " + entrypoint);

    id<MTLComputePipelineState> pipeline =
        [device newComputePipelineStateWithFunction:function error:&error];
    if (pipeline == nil)
      throw std::runtime_error(
          "Metal pipeline creation failed: " +
          std::string([error.localizedDescription UTF8String]));

    const auto maxThreads = std::min<std::size_t>(
        pipeline.maxTotalThreadsPerThreadgroup,
        device.maxThreadsPerThreadgroup.width);
    const auto launch = planMetalLaunch(elementCount, elementCount, output.size(),
                                       options, maxThreads, device.maxBufferLength);
    const std::size_t byteCount = launch.byteCount;
    id<MTLBuffer> inputBuffers[3] = {nil, nil, nil};
    for (std::size_t i = 0; i < inputs.size(); ++i) {
      inputBuffers[i] = [device newBufferWithBytes:inputs[i].data()
                                          length:byteCount
                                         options:MTLResourceStorageModeShared];
      if (inputBuffers[i] == nil)
        throw std::runtime_error("Metal buffer allocation failed");
    }
    id<MTLBuffer> outputBuffer =
        [device newBufferWithLength:byteCount
                            options:MTLResourceStorageModeShared];
    if (outputBuffer == nil)
      throw std::runtime_error("Metal buffer allocation failed");

    id<MTLCommandQueue> queue = [device newCommandQueue];
    id<MTLCommandBuffer> command = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    if (queue == nil || command == nil || encoder == nil)
      throw std::runtime_error("Metal command creation failed");
    [encoder setComputePipelineState:pipeline];
    for (std::size_t i = 0; i < inputs.size(); ++i)
      [encoder setBuffer:inputBuffers[i] offset:0 atIndex:i];
    [encoder setBuffer:outputBuffer offset:0 atIndex:inputs.size()];
    uint32_t count = launch.count;
    [encoder setBytes:&count length:sizeof(count) atIndex:inputs.size() + 1];

    // All groups have the same width, including the padded final group.
    // This matches group_id.x * group_size.x + local_id.x in exported MSL.
    MTLSize groups = MTLSizeMake(launch.threadgroupCount, 1, 1);
    MTLSize group = MTLSizeMake(launch.threadsPerThreadgroup, 1, 1);
    [encoder dispatchThreadgroups:groups threadsPerThreadgroup:group];
    [encoder endEncoding];
    [command commit];
    [command waitUntilCompleted];

    if (command.status != MTLCommandBufferStatusCompleted)
      throw std::runtime_error(
          "Metal execution failed: " +
          std::string([command.error.localizedDescription UTF8String]));

    std::memcpy(output.data(), [outputBuffer contents], byteCount);
  }
}

} // namespace sconeml::tensor
