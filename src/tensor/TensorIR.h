#ifndef SCONEML_TENSOR_IR_H
#define SCONEML_TENSOR_IR_H

#include "mlir/IR/BuiltinOps.h"

#include <memory>
#include <string>

namespace sconeml::tensor {

struct TensorModule {
  std::unique_ptr<mlir::MLIRContext> context;
  mlir::OwningOpRef<mlir::ModuleOp> module;
};

std::string printModule(mlir::Operation *operation);
TensorModule translateTensorProgram(const std::string &source);
std::string translateTensorIR(const std::string &source);

} // namespace sconeml::tensor

#endif
