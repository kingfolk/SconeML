#include "src/tensor/TensorIR.h"

#include "src/dialect/LetAlgDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"

#include "llvm/Support/raw_ostream.h"

#include <stdexcept>

namespace sconeml::tensor {

TensorModule buildPolynomialModule() {
  mlir::DialectRegistry registry;
  mlir::registerAllToLLVMIRTranslations(registry);
  auto context = std::make_unique<mlir::MLIRContext>(registry);
  context->getOrLoadDialect<sconeml::letalg::LetAlgDialect>();
  context->getOrLoadDialect<mlir::arith::ArithDialect>();
  context->getOrLoadDialect<mlir::cf::ControlFlowDialect>();
  context->getOrLoadDialect<mlir::func::FuncDialect>();
  context->getOrLoadDialect<mlir::LLVM::LLVMDialect>();
  context->getOrLoadDialect<mlir::memref::MemRefDialect>();
  context->getOrLoadDialect<mlir::scf::SCFDialect>();

  mlir::OpBuilder builder(context.get());
  mlir::Location loc = builder.getUnknownLoc();
  auto module = mlir::ModuleOp::create(builder, loc);
  builder.setInsertionPointToStart(module.getBody());

  auto bufferType = mlir::MemRefType::get(
      {mlir::ShapedType::kDynamic}, builder.getF32Type());
  auto functionType = builder.getFunctionType({bufferType, bufferType}, {});
  auto function = mlir::func::FuncOp::create(
      builder, loc, "tensor_polynomial", functionType);
  function->setAttr("llvm.emit_c_interface", builder.getUnitAttr());
  mlir::Block *entry = function.addEntryBlock();
  builder.setInsertionPointToStart(entry);

  auto map = sconeml::letalg::TensorMapOp::create(
      builder, loc, entry->getArgument(0), entry->getArgument(1));
  mlir::Block *body = builder.createBlock(
      &map.getBody(), {}, {builder.getF32Type()}, {loc});
  builder.setInsertionPointToStart(body);
  mlir::Value x = body->getArgument(0);
  mlir::Value square = mlir::arith::MulFOp::create(builder, loc, x, x);
  mlir::Value two = mlir::arith::ConstantOp::create(
      builder, loc, builder.getF32FloatAttr(2.0));
  mlir::Value twice = mlir::arith::MulFOp::create(builder, loc, two, x);
  mlir::Value linear =
      mlir::arith::AddFOp::create(builder, loc, square, twice);
  mlir::Value one = mlir::arith::ConstantOp::create(
      builder, loc, builder.getF32FloatAttr(1.0));
  mlir::Value result =
      mlir::arith::AddFOp::create(builder, loc, linear, one);
  sconeml::letalg::YieldOp::create(builder, loc, result);

  builder.setInsertionPointToEnd(entry);
  mlir::func::ReturnOp::create(builder, loc);

  if (mlir::failed(mlir::verify(module)))
    throw std::runtime_error("generated tensor LetAlg module failed verification");
  return {std::move(context), std::move(module)};
}

std::string printModule(mlir::Operation *operation) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  operation->print(stream);
  return text;
}

std::string buildPolynomialIR() {
  TensorModule tensorModule = buildPolynomialModule();
  return printModule(tensorModule.module.get());
}

} // namespace sconeml::tensor
