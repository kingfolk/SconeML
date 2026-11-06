#include "src/tensor/TensorFrontend.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Verifier.h"
#include "src/parser/AstToLetAlg.h"
#include "src/parser/Parser.h"

#include <stdexcept>
#include <limits>
#include <set>

namespace sconeml::tensor {
namespace {
void validateNames(ExprNode *expression, const std::set<std::string> &inputNames) {
  if (expression->getKind() == ExprNode::Kind_Var) {
    auto name = static_cast<VarExprNode *>(expression)->getName();
    if (!inputNames.contains(name))
      throw std::invalid_argument("unknown tensor expression variable: " + name);
  } else if (expression->getKind() == ExprNode::Kind_BinOp) {
    auto *binary = static_cast<BinopExprNode *>(expression);
    validateNames(binary->getL(), inputNames);
    validateNames(binary->getR(), inputNames);
  }
}
} // namespace

void translateTensorExpression(mlir::ModuleOp module,
                               const std::string &source,
                               std::span<const TensorArgumentSpec> arguments) {
  if (arguments.empty() || arguments.size() > 3)
    throw std::invalid_argument("tensor expression requires between one and three tensor arguments");
  std::set<std::string> names;
  for (const auto &argument : arguments) {
    if (!isTensorExpressionIdentifier(argument.name))
      throw std::invalid_argument("invalid tensor argument name: " + argument.name);
    if (!names.insert(argument.name).second)
      throw std::invalid_argument("duplicate tensor argument name: " + argument.name);
    if (argument.elementCount > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
        argument.elementCount > std::numeric_limits<std::size_t>::max() / sizeof(int32_t))
      throw std::invalid_argument("tensor element count exceeds host index/byte range");
    if (argument.elementCount != arguments.front().elementCount)
      throw std::invalid_argument("tensor arguments must have matching lengths");
  }
  auto expression = parseTensorExpression(source);
  validateNames(expression.get(), names);

  mlir::OpBuilder builder(module.getContext());
  auto loc = builder.getUnknownLoc();
  auto buffer = mlir::MemRefType::get(
      {static_cast<int64_t>(arguments.front().elementCount)}, builder.getI32Type());
  builder.setInsertionPointToStart(module.getBody());
  auto function = mlir::func::FuncOp::create(
      builder, loc, "tensor_expression",
      builder.getFunctionType(llvm::SmallVector<mlir::Type>(arguments.size() + 1, buffer), {}));
  function->setAttr("llvm.emit_c_interface", builder.getUnitAttr());
  auto *entry = function.addEntryBlock();
  builder.setInsertionPointToStart(entry);
  sconeml::TranslateContext context{};
  llvm::SmallVector<std::string> orderedNames;
  for (const auto &argument : arguments)
    orderedNames.push_back(argument.name);
  translateTensorMapInto(builder, expression.get(), orderedNames,
                         entry->getArguments().drop_back(),
                         entry->getArgument(arguments.size()), context);
  mlir::func::ReturnOp::create(builder, loc);
  if (mlir::failed(mlir::verify(module)))
    throw std::runtime_error("invalid tensor expression IR");
}

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
