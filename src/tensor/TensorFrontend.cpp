#include "src/tensor/TensorFrontend.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Verifier.h"
#include "src/parser/AstToLetAlg.h"
#include "src/parser/Parser.h"

#include <stdexcept>

namespace sconeml::tensor {

void translateTensorSource(mlir::ModuleOp module, const std::string &source) {
  mlir::OpBuilder builder(module.getContext());
  auto loc = builder.getUnknownLoc();
  builder.setInsertionPointToStart(module.getBody());
  auto function = mlir::func::FuncOp::create(builder, loc, "tensor_literal",
                                             builder.getFunctionType({}, {}));
  function->setAttr("llvm.emit_c_interface", builder.getUnitAttr());
  builder.setInsertionPointToStart(function.addEntryBlock());

  std::string mutableSource = source;
  auto expression = sconeml::parse(mutableSource);
  auto result = sconeml::translate(builder, expression.get());
  function.setFunctionType(builder.getFunctionType({}, {result.getType()}));
  sconeml::letalg::YieldOp::create(builder, loc, result);
  if (mlir::failed(mlir::verify(module)))
    throw std::runtime_error("invalid tensor source IR");
}

} // namespace sconeml::tensor
