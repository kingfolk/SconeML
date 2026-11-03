#ifndef SCONEML_METAL_EXPORTER_H
#define SCONEML_METAL_EXPORTER_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include <string>

namespace sconeml {

struct MetalKernelSource {
  std::string source;
  std::string entrypoint;
};

// Export the verified one-dimensional map kernel produced by LowerTensorToGPU.
// No device APIs are used and the input module is not modified. Unsupported IR
// emits an MLIR diagnostic and returns failure, never partial executable
// source. The emitted ABI is input/output/count at buffer(0/1/2), with uint
// indexing. Launchers must use uniform one-dimensional threadgroups and check
// capacities, count, and padded grid indices against the u32 range before
// execution.
mlir::FailureOr<MetalKernelSource> exportMetalKernel(mlir::ModuleOp module);

} // namespace sconeml

#endif
