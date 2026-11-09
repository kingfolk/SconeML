#ifndef SCONEML_LOWER_SHAPED_TENSOR_TO_GPU_H
#define SCONEML_LOWER_SHAPED_TENSOR_TO_GPU_H
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include "src/tensor/MetalRuntime.h"

namespace sconeml {
// Verify the production Linalg/wrapper envelope, broadcast projections and
// checked per-input/output target counts.
// Throws before device access; empty calls still validate supported shapes.
tensor::MetalKernelLayout validateShapedTensorGPU(mlir::ModuleOp module);
// Extract scalar IR into a guarded GPU map with verified same-index or
// normalized broadcast loads. The source is unchanged; device metadata owns
// input/output shapes and counts.
mlir::FailureOr<mlir::OwningOpRef<mlir::ModuleOp>>
lowerShapedTensorToGPU(mlir::ModuleOp module);
} // namespace sconeml
#endif
