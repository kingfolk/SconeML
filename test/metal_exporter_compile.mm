#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <iostream>

int main(int argc, char **argv) {
  if (argc != 3) {
    std::cerr
        << "usage: metal_exporter_compile <shared.metal> <integer.metal>\n";
    return 1;
  }
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device == nil) {
      std::cout << "SKIP: no Metal device available for exporter compilation\n";
      return 77;
    }
    for (int i = 1; i < argc; ++i) {
      NSError *error = nil;
      NSString *path = [NSString stringWithUTF8String:argv[i]];
      NSString *source = [NSString stringWithContentsOfFile:path
                                                   encoding:NSUTF8StringEncoding
                                                      error:&error];
      if (source == nil) {
        std::cerr << "FAIL: cannot read MSL: " << argv[i] << '\n';
        return 1;
      }
      id<MTLLibrary> library = [device newLibraryWithSource:source
                                                    options:nil
                                                      error:&error];
      if (library == nil) {
        std::cerr << "FAIL: MSL compilation: " <<
            [error.localizedDescription UTF8String] << '\n';
        return 1;
      }
      id<MTLFunction> function = [library newFunctionWithName:@"tensor_map"];
      if (function == nil) {
        std::cerr << "FAIL: missing exported entrypoint\n";
        return 1;
      }
      id<MTLComputePipelineState> pipeline =
          [device newComputePipelineStateWithFunction:function error:&error];
      if (pipeline == nil) {
        std::cerr << "FAIL: pipeline creation: " <<
            [error.localizedDescription UTF8String] << '\n';
        return 1;
      }
    }
    std::cout << "PASS exported MSL compilation and pipelines on "
              << [device.name UTF8String] << " (no dispatch)\n";
  }
  return 0;
}
