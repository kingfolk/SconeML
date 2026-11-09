#include "src/tensor/TensorFrontend.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Verifier.h"
#include "src/parser/AstToLetAlg.h"
#include "src/parser/Parser.h"

#include <algorithm>
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

namespace {
struct DenseShape {
  int64_t count;
  llvm::SmallVector<int64_t> strides;
};
DenseShape checkedDenseShape(llvm::ArrayRef<int64_t> shape) {
  if (shape.empty() || shape.size() > 2)
    throw std::invalid_argument("shaped tensor rank must be 1 or 2");
  for (auto extent : shape)
    if (extent < 0)
      throw std::invalid_argument("shaped tensor extents must be nonnegative");
  DenseShape layout{1, llvm::SmallVector<int64_t>(shape.size())};
  const auto limit = std::min<uint64_t>(
      std::vector<int32_t>().max_size(),
      std::min<uint64_t>(std::numeric_limits<int64_t>::max(),
                         std::numeric_limits<size_t>::max() / sizeof(int32_t)));
  for (size_t axis = shape.size(); axis-- > 0;) {
    layout.strides[axis] = layout.count;
    if (shape[axis] &&
        static_cast<uint64_t>(layout.count) > limit / shape[axis])
      throw std::invalid_argument(
          "shaped tensor exceeds host index/byte/capacity range");
    layout.count *= shape[axis];
  }
  return layout;
}
void shapedRequire(bool valid, const char *message) {
  if (!valid)
    throw std::invalid_argument(std::string("invalid shaped tensor IR: ") +
                                message);
}
} // namespace

void translateShapedTensorExpression(
    mlir::ModuleOp module, const std::string &source,
    std::span<const ShapedTensorArgumentSpec> arguments) {
  if (arguments.empty() || arguments.size() > 3)
    throw std::invalid_argument(
        "shaped expression requires between one and three tensor arguments");
  std::set<std::string> names;
  for (const auto &argument : arguments) {
    if (!isTensorExpressionIdentifier(argument.name))
      throw std::invalid_argument("invalid tensor argument name: " +
                                  argument.name);
    if (!names.insert(argument.name).second)
      throw std::invalid_argument("duplicate tensor argument name: " +
                                  argument.name);
    auto layout = checkedDenseShape(argument.shape);
    if (static_cast<uint64_t>(layout.count) != argument.elementCount)
      throw std::invalid_argument(
          "shaped tensor storage count does not match shape product");
    if (argument.shape != arguments.front().shape)
      throw std::invalid_argument("shaped tensors must have the same shape; "
                                  "broadcasting is not supported yet");
  }
  auto expression = parseTensorExpression(source);
  validateNames(expression.get(), names);
  module.getContext()->getOrLoadDialect<mlir::linalg::LinalgDialect>();
  mlir::OpBuilder builder(module.getContext());
  auto loc = builder.getUnknownLoc();
  const auto &shape = arguments.front().shape;
  const auto layout = checkedDenseShape(shape);
  auto flat = mlir::MemRefType::get({layout.count}, builder.getI32Type());
  auto viewType = mlir::MemRefType::get(
      shape, builder.getI32Type(),
      mlir::StridedLayoutAttr::get(module.getContext(), 0, layout.strides));
  builder.setInsertionPointToStart(module.getBody());
  auto function = mlir::func::FuncOp::create(
      builder, loc, "shaped_tensor_expression",
      builder.getFunctionType(
          llvm::SmallVector<mlir::Type>(arguments.size() + 1, flat), {}));
  function->setAttr("llvm.emit_c_interface", builder.getUnitAttr());
  auto *entry = function.addEntryBlock();
  builder.setInsertionPointToStart(entry);
  llvm::SmallVector<mlir::Value> views;
  for (auto buffer : entry->getArguments())
    views.push_back(mlir::memref::ReinterpretCastOp::create(
        builder, loc, viewType, buffer, 0, shape, layout.strides));
  llvm::SmallVector<mlir::AffineMap> maps(
      arguments.size() + 1, builder.getMultiDimIdentityMap(shape.size()));
  llvm::SmallVector<mlir::utils::IteratorType> iterators(
      shape.size(), mlir::utils::IteratorType::parallel);
  mlir::linalg::GenericOp::create(
      builder, loc, mlir::ValueRange(views).drop_back(),
      mlir::ValueRange{views.back()}, maps, iterators,
      [&](mlir::OpBuilder &body, mlir::Location bodyLoc,
          mlir::ValueRange elements) {
        std::unordered_map<std::string, mlir::Value> bindings;
        for (size_t i = 0; i < arguments.size(); ++i)
          bindings.emplace(arguments[i].name, elements[i]);
        sconeml::TranslateContext context{};
        auto value = translateTensorElementExpr(body, expression.get(),
                                                bindings, context);
        mlir::linalg::YieldOp::create(body, bodyLoc, value);
      });
  mlir::func::ReturnOp::create(builder, loc);
  validateShapedTensorExpression(module);
}

void validateShapedTensorExpression(mlir::ModuleOp module) {
  shapedRequire(mlir::succeeded(mlir::verify(module)),
                "module verification failed");
  shapedRequire(llvm::hasSingleElement(module.getBody()->getOperations()),
                "expected one function");
  auto function =
      mlir::dyn_cast<mlir::func::FuncOp>(&module.getBody()->front());
  shapedRequire(function &&
                    function.getSymName() == "shaped_tensor_expression" &&
                    function->hasAttr("llvm.emit_c_interface") &&
                    function.getNumResults() == 0 &&
                    llvm::hasSingleElement(function.getBody()),
                "expected flat C-interface wrapper");
  auto &entry = function.getBody().front();
  auto arity = entry.getNumArguments();
  shapedRequire(arity >= 2 && arity <= 4, "expected 1-3 inputs and output");
  // Exactly one canonical view per ABI argument, one map and a return.
  shapedRequire(entry.getOperations().size() == arity + 2,
                "unexpected wrapper operations");
  auto op = entry.begin();
  llvm::SmallVector<mlir::Value> views;
  mlir::MemRefType common;
  for (size_t i = 0; i < arity; ++i, ++op) {
    auto view = mlir::dyn_cast<mlir::memref::ReinterpretCastOp>(&*op);
    shapedRequire(view && view.getSource() == entry.getArgument(i),
                  "view operand order");
    auto type = mlir::cast<mlir::MemRefType>(view.getType());
    shapedRequire(type.hasStaticShape() &&
                      type.getElementType().isSignlessInteger(32) &&
                      !type.getMemorySpace(),
                  "expected static host i32 views");
    auto layout = checkedDenseShape(type.getShape());
    auto expected = mlir::MemRefType::get(
        type.getShape(), type.getElementType(),
        mlir::StridedLayoutAttr::get(module.getContext(), 0, layout.strides));
    shapedRequire(type == expected && (!common || type == common),
                  "noncanonical or unequal views");
    common = type;
    shapedRequire(view.getStaticOffsets() == llvm::ArrayRef<int64_t>{0} &&
                      view.getStaticSizes() == type.getShape() &&
                      view.getStaticStrides() ==
                          llvm::ArrayRef<int64_t>(layout.strides) &&
                      view.getOffsets().empty() && view.getSizes().empty() &&
                      view.getStrides().empty(),
                  "expected static zero-offset canonical view");
    auto flat = mlir::MemRefType::get({layout.count}, type.getElementType());
    shapedRequire(entry.getArgument(i).getType() == flat &&
                      entry.getArgument(i).hasOneUse(),
                  "flat buffer count/type/use mismatch");
    views.push_back(view.getResult());
  }
  auto map = mlir::dyn_cast<mlir::linalg::GenericOp>(&*op++);
  shapedRequire(map && map.getNumResults() == 0 &&
                    map.getInputs().size() == arity - 1 &&
                    map.getOutputs().size() == 1 && !map.getLibraryCall(),
                "expected one buffer map");
  shapedRequire(llvm::equal(map->getOperands(), views),
                "map buffer order mismatch");
  for (auto view : views)
    shapedRequire(view.hasOneUse(), "view must only be used by the map");
  auto maps = map.getIndexingMapsArray();
  shapedRequire(maps.size() == arity, "indexing map count mismatch");
  for (auto indexing : maps)
    shapedRequire(indexing == mlir::AffineMap::getMultiDimIdentityMap(
                                  common.getRank(), module.getContext()),
                  "only identity indexing is supported");
  auto iterators = map.getIteratorTypesArray();
  shapedRequire(iterators.size() == static_cast<size_t>(common.getRank()) &&
                    llvm::all_of(iterators,
                                 [](auto iterator) {
                                   return iterator ==
                                          mlir::utils::IteratorType::parallel;
                                 }),
                "only parallel iterators are supported");
  shapedRequire(llvm::hasSingleElement(map.getRegion()),
                "expected single scalar block");
  auto &body = map.getRegion().front();
  shapedRequire(body.getNumArguments() == arity,
                "scalar argument count mismatch");
  for (auto argument : body.getArguments())
    shapedRequire(argument.getType().isSignlessInteger(32),
                  "expected i32 scalar arguments");
  shapedRequire(body.getArguments().back().use_empty(),
                "map must not read uninitialized output");
  for (auto &scalar : body) {
    for (auto operand : scalar.getOperands())
      shapedRequire(operand.getParentBlock() == &body,
                    "scalar captures are not supported");
    if (mlir::isa<mlir::linalg::YieldOp>(scalar))
      continue;
    shapedRequire(
        mlir::isMemoryEffectFree(&scalar) &&
            mlir::isa<mlir::arith::ConstantOp, mlir::arith::AddIOp,
                      mlir::arith::SubIOp, mlir::arith::MulIOp>(scalar) &&
            scalar.getNumResults() == 1 &&
            scalar.getResult(0).getType().isSignlessInteger(32),
        "scalar body supports only i32 constants and add/sub/mul");
    if (auto flags =
            mlir::dyn_cast<mlir::arith::ArithIntegerOverflowFlagsInterface>(
                &scalar))
      shapedRequire(!flags.hasNoSignedWrap() && !flags.hasNoUnsignedWrap(),
                    "scalar arithmetic must use wrapping i32 semantics");
  }
  shapedRequire(mlir::isa<mlir::func::ReturnOp>(&*op),
                "expected wrapper return");
}

} // namespace sconeml::tensor
