#ifndef SCONEML_TENSOR_FRONTEND_H
#define SCONEML_TENSOR_FRONTEND_H

#include "mlir/IR/BuiltinOps.h"
#include <string>

namespace sconeml::tensor {
// Populate a module using the same parser and translation as the CPU runner.
void translateTensorSource(mlir::ModuleOp module, const std::string &source);
}

#endif
