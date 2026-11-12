#include "src/tensor/TensorFrontend.h"
#include "src/tensor/F32Support.h"

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
// Right alignment is deliberate: [R] aligns to columns, never to rows.
// In particular, 0 broadcast with 1 stays 0; numerical max would be wrong.
std::vector<int64_t>
inferBroadcastShape(llvm::ArrayRef<llvm::ArrayRef<int64_t>> shapes) {
  size_t rank = 1;
  for (auto shape : shapes)
    rank = std::max(rank, shape.size());
  std::vector<int64_t> result(rank, 1);
  for (auto shape : shapes)
    for (size_t axis = 0; axis < shape.size(); ++axis) {
      auto &extent = result[rank - shape.size() + axis];
      if (extent == shape[axis])
        continue;
      if (extent == 1)
        extent = shape[axis];
      else if (shape[axis] != 1)
        throw std::invalid_argument(
            "incompatible shaped tensor broadcast dimensions");
    }
  // Individually small inputs can produce an unrepresentable outer product.
  checkedDenseShape(result);
  return result;
}

mlir::AffineMap broadcastMap(llvm::ArrayRef<int64_t> inputShape,
                             llvm::ArrayRef<int64_t> outputShape,
                             mlir::MLIRContext *context) {
  llvm::SmallVector<mlir::AffineExpr> indices;
  for (size_t axis = 0; axis < inputShape.size(); ++axis)
    indices.push_back(inputShape[axis] == 1
                          ? mlir::getAffineConstantExpr(0, context)
                          : mlir::getAffineDimExpr(outputShape.size() -
                                                       inputShape.size() + axis,
                                                   context));
  return mlir::AffineMap::get(outputShape.size(), 0, indices, context);
}

void shapedRequire(bool valid, const char *message) {
  if (!valid)
    throw std::invalid_argument(std::string("invalid shaped tensor IR: ") +
                                message);
}
} // namespace

namespace {
mlir::Value translateF32Element(
    mlir::OpBuilder &builder, ExprNode *expression,
    const std::unordered_map<std::string, mlir::Value> &bindings) {
  auto loc = builder.getUnknownLoc();
  if (expression->getKind() == ExprNode::Kind_Var)
    return bindings.at(static_cast<VarExprNode *>(expression)->getName());
  if (expression->getKind() == ExprNode::Kind_Float)
    return mlir::arith::ConstantOp::create(
        builder, loc,
        builder.getFloatAttr(builder.getF32Type(),
                             parseF32Literal(expression->dump())));
  auto *binary = static_cast<BinopExprNode *>(expression);
  auto left = translateF32Element(builder, binary->getL(), bindings);
  auto right = translateF32Element(builder, binary->getR(), bindings);
  mlir::Value value;
  if (binary->getOp() == '*')
    value = mlir::arith::MulFOp::create(builder, loc, left, right);
  else if (binary->getOp() == '-')
    value = mlir::arith::SubFOp::create(builder, loc, left, right);
  else
    value = mlir::arith::AddFOp::create(builder, loc, left, right);
  return binary->getOp() == '*'
             ? flushF32Product(builder, loc, value, left, right)
             : flushF32(builder, loc, value);
}

std::vector<int64_t>
translateShaped(mlir::ModuleOp module, const std::string &source,
                std::span<const ShapedTensorArgumentSpec> arguments, bool f32) {

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
  }
  llvm::SmallVector<llvm::ArrayRef<int64_t>> inputShapes;
  for (const auto &argument : arguments)
    inputShapes.push_back(argument.shape);
  const auto shape = inferBroadcastShape(inputShapes);
  auto expression =
      f32 ? parseF32TensorExpression(source) : parseTensorExpression(source);
  validateNames(expression.get(), names);
  module.getContext()->getOrLoadDialect<mlir::linalg::LinalgDialect>();
  mlir::OpBuilder builder(module.getContext());
  auto loc = builder.getUnknownLoc();
  mlir::Type elementType =
      f32 ? mlir::Type(builder.getF32Type()) : mlir::Type(builder.getI32Type());
  llvm::SmallVector<mlir::Type> flatTypes;
  llvm::SmallVector<llvm::ArrayRef<int64_t>> viewShapes(inputShapes);
  viewShapes.push_back(shape);
  for (auto viewShape : viewShapes)
    flatTypes.push_back(mlir::MemRefType::get(
        {checkedDenseShape(viewShape).count}, elementType));
  builder.setInsertionPointToStart(module.getBody());
  auto function =
      mlir::func::FuncOp::create(builder, loc, "shaped_tensor_expression",
                                 builder.getFunctionType(flatTypes, {}));
  function->setAttr("llvm.emit_c_interface", builder.getUnitAttr());
  if (f32)
    function->setAttr("sconeml.float_profile",
                      builder.getStringAttr("f32-ftz"));
  auto *entry = function.addEntryBlock();
  builder.setInsertionPointToStart(entry);
  llvm::SmallVector<mlir::Value> views;
  for (auto [buffer, viewShape] :
       llvm::zip(entry->getArguments(), viewShapes)) {
    auto layout = checkedDenseShape(viewShape);
    auto viewType = mlir::MemRefType::get(
        viewShape, elementType,
        mlir::StridedLayoutAttr::get(module.getContext(), 0, layout.strides));
    views.push_back(mlir::memref::ReinterpretCastOp::create(
        builder, loc, viewType, buffer, 0, viewShape, layout.strides));
  }
  llvm::SmallVector<mlir::AffineMap> maps;
  for (auto inputShape : inputShapes)
    maps.push_back(broadcastMap(inputShape, shape, module.getContext()));
  maps.push_back(builder.getMultiDimIdentityMap(shape.size()));
  llvm::SmallVector<mlir::utils::IteratorType> iterators(
      shape.size(), mlir::utils::IteratorType::parallel);
  mlir::linalg::GenericOp::create(
      builder, loc, mlir::ValueRange(views).drop_back(),
      mlir::ValueRange{views.back()}, maps, iterators,
      [&](mlir::OpBuilder &body, mlir::Location bodyLoc,
          mlir::ValueRange elements) {
        std::unordered_map<std::string, mlir::Value> bindings;
        for (size_t i = 0; i < arguments.size(); ++i)
          bindings.emplace(arguments[i].name,
                           f32 ? flushF32(body, bodyLoc, elements[i])
                               : elements[i]);
        sconeml::TranslateContext context{};
        auto value = f32 ? translateF32Element(body, expression.get(), bindings)
                         : translateTensorElementExpr(body, expression.get(),
                                                      bindings, context);
        mlir::linalg::YieldOp::create(body, bodyLoc, value);
      });
  mlir::func::ReturnOp::create(builder, loc);
  validateShapedTensorExpression(module);
  return shape;
}

} // namespace
std::vector<int64_t> translateShapedTensorExpression(
    mlir::ModuleOp module, const std::string &source,
    std::span<const ShapedTensorArgumentSpec> arguments) {
  return translateShaped(module, source, arguments, false);
}
std::vector<int64_t> translateShapedTensorF32Expression(
    mlir::ModuleOp module, const std::string &source,
    std::span<const ShapedTensorArgumentSpec> arguments) {
  return translateShaped(module, source, arguments, true);
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
  auto firstBuffer = mlir::dyn_cast<mlir::MemRefType>(entry.getArgument(0).getType());
  shapedRequire(static_cast<bool>(firstBuffer), "expected flat memref arguments");
  auto elementType = firstBuffer.getElementType();
  const bool f32 = elementType.isF32();
  auto profile =
      function->getAttrOfType<mlir::StringAttr>("sconeml.float_profile");
  shapedRequire(f32 ? (profile && profile.getValue() == "f32-ftz")
                    : !function->hasAttr("sconeml.float_profile"),
                "unsupported floating-point profile");
  // Exactly one canonical view per ABI argument, one map and a return.
  shapedRequire(entry.getOperations().size() == arity + 2,
                "unexpected wrapper operations");
  auto op = entry.begin();
  llvm::SmallVector<mlir::Value> views;
  llvm::SmallVector<llvm::ArrayRef<int64_t>> inputShapes;
  llvm::ArrayRef<int64_t> outputShape;
  for (size_t i = 0; i < arity; ++i, ++op) {
    auto view = mlir::dyn_cast<mlir::memref::ReinterpretCastOp>(&*op);
    shapedRequire(view && view.getSource() == entry.getArgument(i),
                  "view operand order");
    auto type = mlir::cast<mlir::MemRefType>(view.getType());
    shapedRequire(type.hasStaticShape() &&
                      (type.getElementType().isSignlessInteger(32) || f32) &&
                      type.getElementType() == elementType &&
                      !type.getMemorySpace(),
                  "expected homogeneous static host i32/f32 views");
    auto layout = checkedDenseShape(type.getShape());
    auto expected = mlir::MemRefType::get(
        type.getShape(), type.getElementType(),
        mlir::StridedLayoutAttr::get(module.getContext(), 0, layout.strides));
    shapedRequire(type == expected, "noncanonical view");
    if (i + 1 == arity)
      outputShape = type.getShape();
    else
      inputShapes.push_back(type.getShape());
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
  const auto inferred = inferBroadcastShape(inputShapes);
  shapedRequire(outputShape == llvm::ArrayRef<int64_t>(inferred),
                "output shape does not match inferred broadcast shape");
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
  for (size_t i = 0; i + 1 < arity; ++i)
    shapedRequire(maps[i] == broadcastMap(inputShapes[i], outputShape,
                                          module.getContext()),
                  "input indexing must match inferred broadcast map");
  shapedRequire(maps.back() == mlir::AffineMap::getMultiDimIdentityMap(
                                   outputShape.size(), module.getContext()),
                "output indexing must be identity");
  auto iterators = map.getIteratorTypesArray();
  shapedRequire(iterators.size() == outputShape.size() &&
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
    shapedRequire(argument.getType() == elementType,
                  "expected matching scalar arguments");
  shapedRequire(body.getArguments().back().use_empty(),
                "map must not read uninitialized output");
  if (f32) {
    auto yield = mlir::dyn_cast<mlir::linalg::YieldOp>(body.getTerminator());
    shapedRequire(
        yield && yield.getValues().size() == 1 &&
            validateF32Scalars(body.begin(), std::prev(body.end()),
                               body.getArguments().drop_back(),
                               yield.getValues().front()),
        "f32-ftz body requires normalized inputs/results and no fast-math");
  } else
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
