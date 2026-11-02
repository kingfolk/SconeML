#ifndef SCONEML_TENSOR_FRONTEND_H
#define SCONEML_TENSOR_FRONTEND_H

#include "mlir/IR/BuiltinOps.h"
#include <string>

namespace sconeml::tensor {
// Populate and verify a module using the shared CPU/GPU frontend.
// Throws on translation or verification failure; no target code is executed.
void translateTensorSource(mlir::ModuleOp module, const std::string &source);
}

#endif
