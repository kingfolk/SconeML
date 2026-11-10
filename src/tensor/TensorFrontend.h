#ifndef SCONEML_TENSOR_FRONTEND_H
#define SCONEML_TENSOR_FRONTEND_H

#include "mlir/IR/BuiltinOps.h"
#include <string>
#include <span>
#include <vector>

namespace sconeml::tensor {
// Populate and verify a module using the shared CPU/GPU frontend.
// Throws on translation or verification failure; no target code is executed.
void translateTensorSource(mlir::ModuleOp module, const std::string &source);

// Shape/name information only: host element values never enter translation.
struct TensorArgumentSpec {
  std::string name;
  std::size_t elementCount;
};

// Translate constants, 1-3 named tensors, and wrapping + / - / * to a function
// with explicit input/output buffers. Specializes the shape for this call.
// Arguments must have unique names and matching lengths. Empty shapes are valid;
// names and expression syntax are still validated. No execution occurs here.
void translateTensorExpression(mlir::ModuleOp module,
                               const std::string &expression,
                               std::span<const TensorArgumentSpec> arguments);
struct ShapedTensorArgumentSpec {
  std::string name;
  std::size_t elementCount;
  std::vector<int64_t> shape;
};
// Infers trailing-axis broadcasting over every declared input, validates names,
// shapes and storage, and returns the checked output shape. Builds flat C-ABI
// buffers, canonical views and linalg.generic with broadcast indexing maps.
std::vector<int64_t> translateShapedTensorExpression(
    mlir::ModuleOp module, const std::string &expression,
    std::span<const ShapedTensorArgumentSpec> arguments);
// Homogeneous f32 with explicit signed flush-to-zero at every scalar boundary.
std::vector<int64_t> translateShapedTensorF32Expression(
    mlir::ModuleOp module, const std::string &expression,
    std::span<const ShapedTensorArgumentSpec> arguments);
// Narrow production envelope validation, also used before CPU lowering.
// Arbitrary Linalg, captures, output reads and noncanonical views are rejected.
void validateShapedTensorExpression(mlir::ModuleOp module);
}

#endif
