#include "src/tensor/GPURunner.h"

#include "src/dialect/LetAlgDialect.h"
#include "src/tensor/MetalRuntime.h"
#include "src/tensor/TensorIR.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"

#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace sconeml::tensor {
namespace {

std::string emitMetalExpression(mlir::Value value,
                                mlir::BlockArgument scalarArgument) {
  if (value == scalarArgument)
    return "x";

  mlir::Operation *definition = value.getDefiningOp();
  if (!definition)
    throw std::runtime_error("unsupported free value in tensor map body");

  if (auto constant = mlir::dyn_cast<mlir::arith::ConstantOp>(definition)) {
    auto attribute = mlir::dyn_cast<mlir::FloatAttr>(constant.getValue());
    if (!attribute)
      throw std::runtime_error("Metal tensor map only supports f32 constants");
    std::ostringstream stream;
    stream << std::showpoint << std::setprecision(9)
           << attribute.getValueAsDouble() << "f";
    return stream.str();
  }

  auto binary = [&](const char *symbol) {
    return "(" + emitMetalExpression(definition->getOperand(0), scalarArgument) +
           " " + symbol + " " +
           emitMetalExpression(definition->getOperand(1), scalarArgument) +
           ")";
  };
  if (mlir::isa<mlir::arith::AddFOp>(definition))
    return binary("+");
  if (mlir::isa<mlir::arith::SubFOp>(definition))
    return binary("-");
  if (mlir::isa<mlir::arith::MulFOp>(definition))
    return binary("*");
  if (mlir::isa<mlir::arith::DivFOp>(definition))
    return binary("/");

  throw std::runtime_error("unsupported operation in Metal tensor map: " +
                           definition->getName().getStringRef().str());
}

std::string translateTensorMapToMetal(mlir::ModuleOp module) {
  sconeml::letalg::TensorMapOp map;
  module.walk([&](sconeml::letalg::TensorMapOp candidate) { map = candidate; });
  if (!map)
    throw std::runtime_error("module does not contain letalg.tensor_map");

  mlir::Block &body = map.getBody().front();
  auto yield = mlir::cast<sconeml::letalg::YieldOp>(body.getTerminator());
  std::string expression =
      emitMetalExpression(yield.getExpr(), body.getArgument(0));

  return R"metal(#include <metal_stdlib>
using namespace metal;

kernel void tensor_map(device const float *input [[buffer(0)]],
                       device float *output [[buffer(1)]],
                       constant uint &count [[buffer(2)]],
                       uint index [[thread_position_in_grid]]) {
  if (index >= count) return;
  float x = input[index];
  output[index] = )metal" + expression + R"metal(;
}
)metal";
}

} // namespace

std::vector<float> runPolynomialGPU(const std::vector<float> &input,
                                    std::string *gpuSource) {
  TensorModule tensorModule = buildPolynomialModule();
  std::string source = translateTensorMapToMetal(*tensorModule.module);
  if (gpuSource)
    *gpuSource = source;
  std::vector<float> output(input.size(), 0.0f);
  runMetalKernel(source, input.data(), output.data(), input.size());
  return output;
}

bool gpuIsAvailable() { return metalIsAvailable(); }
std::string gpuName() { return metalDeviceName(); }

} // namespace sconeml::tensor
