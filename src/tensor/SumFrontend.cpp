#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "src/tensor/Sum.h"
#include <algorithm>
#include <charconv>
#include <limits>
#include <stdexcept>

namespace sconeml::tensor {
namespace {
std::string trim(std::string_view s) {
  auto a = s.find_first_not_of(" \t\r\n");
  if (a == s.npos)
    return {};
  return std::string(s.substr(a, s.find_last_not_of(" \t\r\n") - a + 1));
}
size_t count(std::span<const int64_t> shape) {
  size_t n = 1;
  for (auto d : shape) {
    if (d < 0 ||
        static_cast<uint64_t>(d) > std::numeric_limits<uint32_t>::max() ||
        (d &&
         n > std::numeric_limits<uint32_t>::max() / static_cast<uint64_t>(d)))
      throw std::invalid_argument(
          "sum shape/count exceeds nonnegative u32 range");
    n *= d;
  }
  return n;
}
} // namespace
SumExpression parseSumExpression(const std::string &source) {
  auto s = trim(source);
  auto open = s.find('(');
  if (open == s.npos || trim(std::string_view(s).substr(0, open)) != "sum" ||
      s.back() != ')')
    throw std::invalid_argument("sum requires sum(expression[, axis=N])");
  auto body = std::string_view(s).substr(open + 1, s.size() - open - 2);
  auto comma = body.find(',');
  SumExpression result{trim(body.substr(0, comma)), {}};
  if (result.map.empty())
    throw std::invalid_argument("sum requires an operand");
  if (comma != body.npos) {
    auto suffix = trim(body.substr(comma + 1));
    auto equal = suffix.find('=');
    if (equal == suffix.npos ||
        trim(std::string_view(suffix).substr(0, equal)) != "axis")
      throw std::invalid_argument(
          "sum supports only one nonnegative axis; keepdims is unsupported");
    auto text = trim(std::string_view(suffix).substr(equal + 1));
    int64_t axis = -1;
    auto [end, error] =
        std::from_chars(text.data(), text.data() + text.size(), axis);
    if (error != std::errc{} || end != text.data() + text.size() || axis < 0)
      throw std::invalid_argument("sum axis must be one nonnegative integer");
    result.axis = axis;
  }
  return result;
}
SumPlan makeSumPlan(std::span<const int64_t> shape,
                    std::optional<int64_t> axis) {
  if (shape.empty() || shape.size() > 2)
    throw std::invalid_argument("sum input rank must be 1 or 2");
  if (axis && static_cast<uint64_t>(*axis) >= shape.size())
    throw std::invalid_argument("sum axis is outside input rank");
  SumPlan p;
  p.inputShape.assign(shape.begin(), shape.end());
  p.axis = axis;
  p.inputCount = count(shape);
  p.length = p.inputCount;
  p.outputCount = 1;
  p.baseStride = 0;
  if (axis) {
    p.length = shape[*axis];
    p.outputShape = p.inputShape;
    p.outputShape.erase(p.outputShape.begin() + *axis);
    p.outputCount = count(p.outputShape);
    if (shape.size() == 2) {
      p.baseStride = *axis == 0 ? 1 : shape[1];
      p.inputStride = *axis == 0 ? shape[1] : 1;
    }
  }
  // Padding/scratch sizes are bounded before allocation, including empty cases.
  auto groups = p.length / 256 + (p.length % 256 != 0);
  if (groups &&
      p.outputCount > std::numeric_limits<uint32_t>::max() / groups / 256)
    throw std::invalid_argument("sum padded scratch/index range exceeds u32");
  return p;
}
void validateSumPlan(const SumPlan &p) {
  if (p != makeSumPlan(p.inputShape, p.axis))
    throw std::invalid_argument("sum metadata does not match shape/axis");
}
SumPlan prepareSum(const std::string &source,
                   std::span<const ShapedTensorArgumentSpec> args, bool f32) {
  auto expression = parseSumExpression(source);
  if (args.size() != 1 || expression.map != args.front().name)
    throw std::invalid_argument("sum currently requires one named input; map "
                                "composition follows in M5-C");
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::arith::ArithDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::memref::MemRefDialect>();
  context.getOrLoadDialect<mlir::linalg::LinalgDialect>();
  auto m = mlir::OwningOpRef<mlir::ModuleOp>(
      mlir::ModuleOp::create(mlir::UnknownLoc::get(&context)));
  auto shape =
      f32 ? translateShapedTensorF32Expression(*m, expression.map, args)
          : translateShapedTensorExpression(*m, expression.map, args);
  return makeSumPlan(shape, expression.axis);
}
} // namespace sconeml::tensor
