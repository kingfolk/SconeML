#ifndef SCONEML_LOWER_TENSOR_TO_GPU_H
#define SCONEML_LOWER_TENSOR_TO_GPU_H

#include "mlir/Support/LogicalResult.h"
#include "src/dialect/LetAlgDialect.h"

namespace sconeml {

// Extract one verified map into a new device-only module containing
// gpu.module @kernels / gpu.func @tensor_map. The source is not modified.
// The caller must keep the source MLIRContext alive for the returned module.
//
// Kernel arguments: global contiguous i32 input/output memrefs, then an i32
// count interpreted as unsigned. The eventual launcher must check capacities,
// distinct allocations and the u32 range of count and padded grid indices.
// This function neither generates a host launch nor executes the kernel.
mlir::FailureOr<mlir::OwningOpRef<mlir::ModuleOp>>
lowerTensorMapToGPU(letalg::TensorMapOp map);

} // namespace sconeml

#endif
