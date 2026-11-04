#include "test/runner/Pipelines.h"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdexcept>

namespace sconeml::test {
void compileMetal(const std::string &source, const std::string &entrypoint) {
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    require(device != nil, "Metal device unavailable");
    NSError *error = nil;
    id<MTLLibrary> library = [device
        newLibraryWithSource:[NSString stringWithUTF8String:source.c_str()]
                     options:nil
                       error:&error];
    if (library == nil)
      throw std::runtime_error(std::string("Metal compilation failed: ") +
                               [error.localizedDescription UTF8String]);
    id<MTLFunction> function = [library
        newFunctionWithName:[NSString stringWithUTF8String:entrypoint.c_str()]];
    require(function != nil, "exported Metal entrypoint not found");
    id<MTLComputePipelineState> pipeline =
        [device newComputePipelineStateWithFunction:function error:&error];
    if (pipeline == nil)
      throw std::runtime_error(std::string("Metal pipeline failed: ") +
                               [error.localizedDescription UTF8String]);
  }
}
} // namespace sconeml::test
