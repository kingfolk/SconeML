#include "src/tensor/SumGPU.h"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstring>
#include <stdexcept>
namespace sconeml::tensor {
namespace {
std::runtime_error failure(const char *stage, NSError *error) {
  return std::runtime_error(
      std::string(stage) + ": " +
      (error ? error.localizedDescription.UTF8String : "unknown Metal error"));
}
} // namespace
void detail::runSumMetalBytes(const SumMetalProgram &p, MetalElementType type,
                              std::span<const std::byte> input,
                              std::span<std::byte> output) {
  if (input.size() % 4 || output.size() % 4)
    throw std::invalid_argument("sum buffer bytes are not whole elements");
  validateSumMetal(p, type, input.size() / 4, output.size() / 4);
  if ((!input.empty() && !input.data()) || (!output.empty() && !output.data()))
    throw std::invalid_argument(
        "sum nonempty buffers require nonnull pointers");
  if (p.dispatches.empty()) {
    if (!output.empty())
      std::memset(output.data(), 0, output.size());
    return;
  }
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device)
      throw std::runtime_error("no Metal GPU is available for sum");
    validateSumMetal(p, type, input.size() / 4, output.size() / 4,
                     device.maxBufferLength);
    MTLCompileOptions *options = [MTLCompileOptions new];
    if (@available(macOS 15.0, *)) {
      options.mathMode = MTLMathModeSafe;
      options.mathFloatingPointFunctions = MTLMathFloatingPointFunctionsPrecise;
    } else
      options.fastMathEnabled = NO;
    NSError *error = nil;
    auto library = [device
        newLibraryWithSource:[NSString stringWithUTF8String:p.source.c_str()]
                     options:options
                       error:&error];
    if (!library)
      throw failure("sum Metal compilation", error);
    auto function = [library newFunctionWithName:@"sum_tree"];
    if (!function)
      throw std::runtime_error("sum Metal entrypoint missing");
    auto pipeline = [device newComputePipelineStateWithFunction:function
                                                          error:&error];
    if (!pipeline)
      throw failure("sum Metal pipeline", error);
    if (pipeline.maxTotalThreadsPerThreadgroup < 256 ||
        device.maxThreadsPerThreadgroup.width < 256 ||
        pipeline.staticThreadgroupMemoryLength >
            device.maxThreadgroupMemoryLength)
      throw std::runtime_error(
          "sum device does not support required 256-lane scratch/threadgroup");
    id<MTLBuffer> current =
        [device newBufferWithBytes:input.data()
                            length:input.size()
                           options:MTLResourceStorageModeShared];
    if (!current)
      throw std::runtime_error("sum input buffer allocation failed");
    __attribute__((
        objc_precise_lifetime)) NSMutableArray<id<MTLBuffer>> *retained =
        [NSMutableArray new];
    [retained addObject:current];
    auto queue = [device newCommandQueue];
    auto command = [queue commandBuffer];
    if (!queue || !command)
      throw std::runtime_error("sum command creation failed");
    // Ordinary tracked buffers and separate serial compute encoders establish
    // producer/consumer dependencies. No barrier crosses threadgroups.
    for (const auto &d : p.dispatches) {
      auto next = [device newBufferWithLength:d.groupCount * 4
                                      options:MTLResourceStorageModeShared];
      if (!next)
        throw std::runtime_error("sum partial buffer allocation failed");
      [retained addObject:next];
      auto encoder = [command computeCommandEncoder];
      if (!encoder)
        throw std::runtime_error("sum encoder creation failed");
      [encoder setComputePipelineState:pipeline];
      [encoder setBuffer:current offset:0 atIndex:0];
      [encoder setBuffer:next offset:0 atIndex:1];
      uint32_t params[] = {static_cast<uint32_t>(d.length),
                           static_cast<uint32_t>(d.parts),
                           static_cast<uint32_t>(d.baseStride),
                           static_cast<uint32_t>(d.inputStride)};
      for (unsigned i = 0; i < 4; ++i)
        [encoder setBytes:&params[i] length:sizeof(uint32_t) atIndex:i + 2];
      [encoder dispatchThreadgroups:MTLSizeMake(d.groupCount, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      [encoder endEncoding];
      current = next;
    }
    [command commit];
    [command waitUntilCompleted];
    if (command.status != MTLCommandBufferStatusCompleted)
      throw failure("sum Metal execution", command.error);
    std::memcpy(output.data(), current.contents, output.size());
    (void)retained; // Own every partial until all encoders complete.
  }
}
} // namespace sconeml::tensor
