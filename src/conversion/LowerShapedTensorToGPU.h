#ifndef SCONEML_LOWER_SHAPED_TENSOR_TO_GPU_H
#define SCONEML_LOWER_SHAPED_TENSOR_TO_GPU_H
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include "src/tensor/MetalRuntime.h"

namespace sconeml {
// Verify the production Linalg/wrapper envelope and target bounds. M3-C allows
// identical shapes only, including semantically identity singleton axes.
// Throws before device access; empty calls still validate supported shapes.
tensor::MetalKernelLayout validateShapedTensorGPU(mlir::ModuleOp module);
// Extract scalar IR through the existing verified TensorMap device path. Does
// not modify the host module. Shape/count metadata is attached to device IR.
mlir::FailureOr<mlir::OwningOpRef<mlir::ModuleOp>>
lowerShapedTensorToGPU(mlir::ModuleOp module);
} // namespace sconeml
#endif
