#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#pragma STDC FP_CONTRACT OFF
int main() {
  @autoreleasepool {
    id<MTLDevice> d = MTLCreateSystemDefaultDevice();
    if (!d) {
      puts("no Metal device");
      return 1;
    }
    printf("device=%s\n", d.name.UTF8String);
    const uint32_t bits[][3] = {
        {0x3f800001, 0x3f7ffffe, 0xbf800000}, // separated mul/add 0; fma -2^-46
        {0x3f800000, 0x33800000, 0x00000000}, // RNE half ulp
        {0x3f800001, 0x33800000, 0x00000000}, // odd tie
        {0x00800000, 0x3f000000, 0x00000000}, // subnormal product
        {0x00000001, 0x40000000, 0x00000000}, // subnormal operand
        {0x80000000, 0x3f800000, 0x80000000},
        {0x7f800000, 0x00000000, 0xff800000},
        {0x7fc00001, 0x3f800000, 0x3f800000},
        {0x7f7fffff, 0x40000000, 0x00000000},
        {0x00800001, 0x00800000, 0x00000000},
        {0x00800000, 0x3f7fffff,
         0x00000000}, // tiny exact product rounding to minimum normal
        {0x80800000, 0x3f7fffff, 0x80000000},
        {0x00800001, 0x3f7ffffe, 0x00000000},
        {0x3f800000, 0x00000001, 0x00000000}};
    constexpr size_t n = sizeof(bits) / sizeof(bits[0]);
    float values[3 * n];
    memcpy(values, bits, sizeof(bits));
    NSString *s = @"#include <metal_stdlib>\nusing namespace metal;\n#pragma "
                  @"STDC FP_CONTRACT OFF\nkernel void probe(device const float "
                  @"*in [[buffer(0)]], device float *out [[buffer(1)]],uint i "
                  @"[[thread_position_in_grid]]){float "
                  @"a=in[3*i],b=in[3*i+1],c=in[3*i+2];out[4*i]=a+b;out[4*i+1]="
                  @"a-b;out[4*i+2]=a*b;float p=a*b;out[4*i+3]=p+c;}";
    MTLCompileOptions *o = [MTLCompileOptions new];
    o.fastMathEnabled = NO;
    if (@available(macOS 15.0, *)) {
      o.mathMode = MTLMathModeSafe;
      o.mathFloatingPointFunctions = MTLMathFloatingPointFunctionsPrecise;
    }
    NSError *e = nil;
    id<MTLLibrary> l = [d newLibraryWithSource:s options:o error:&e];
    if (!l) {
      puts(e.description.UTF8String);
      return 2;
    }
    id<MTLComputePipelineState> p =
        [d newComputePipelineStateWithFunction:[l newFunctionWithName:@"probe"]
                                         error:&e];
    if (!p) {
      puts(e.description.UTF8String);
      return 3;
    }
    id<MTLBuffer> in = [d newBufferWithBytes:values
                                      length:sizeof(values)
                                     options:MTLResourceStorageModeShared];
    id<MTLBuffer> out = [d newBufferWithLength:4 * n * sizeof(float)
                                       options:MTLResourceStorageModeShared];
    id<MTLCommandQueue> q = [d newCommandQueue];
    id<MTLCommandBuffer> cb = [q commandBuffer];
    id<MTLComputeCommandEncoder> en = [cb computeCommandEncoder];
    [en setComputePipelineState:p];
    [en setBuffer:in offset:0 atIndex:0];
    [en setBuffer:out offset:0 atIndex:1];
    [en dispatchThreads:MTLSizeMake(n, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [en endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.status != MTLCommandBufferStatusCompleted) {
      puts(cb.error.description.UTF8String);
      return 4;
    }
    auto *gpu = (uint32_t *)out.contents;
    for (int i = 0; i < n; ++i) {
      volatile float a = values[3 * i], b = values[3 * i + 1],
                     c = values[3 * i + 2];
      volatile float prod = a * b;
      float cpu[4] = {a + b, a - b, prod, prod + c};
      uint32_t u[4];
      memcpy(u, cpu, sizeof(u));
      printf("%d cpu %08x %08x %08x %08x gpu %08x %08x %08x %08x\n", i, u[0],
             u[1], u[2], u[3], gpu[4 * i], gpu[4 * i + 1], gpu[4 * i + 2],
             gpu[4 * i + 3]);
    }
  }
}
