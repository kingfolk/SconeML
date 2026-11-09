#ifndef SCONEML_METAL_EXPORTER_H
#define SCONEML_METAL_EXPORTER_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include <string>
#include <cstddef>
#include <optional>
#include "src/tensor/MetalRuntime.h"

namespace sconeml {

struct MetalKernelSource {
  std::string source;
  std::string entrypoint;
  std::size_t inputCount = 1;
  std::optional<tensor::MetalKernelLayout> shapedLayout;
};

// Export a verified legacy/same-shape or normalized broadcast GPU map kernel.
// No device APIs are used and the input module is not modified. Unsupported IR
// emits an MLIR diagnostic and returns failure, never partial executable
// source. For K inputs (1-3), bindings are inputs 0..K-1, output K, count K+1;
// inputCount reports the validated arity. Indexing uses uint. Launchers must use
// uniform one-dimensional threadgroups and check
// capacities, count, and padded grid indices against the u32 range before
// execution. Broadcast metadata additionally proves per-input indices/counts;
// div/rem is accepted only on that checked index path.
mlir::FailureOr<MetalKernelSource> exportMetalKernel(mlir::ModuleOp module);

} // namespace sconeml

#endif
