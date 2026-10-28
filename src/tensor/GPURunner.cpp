#include "src/tensor/GPURunner.h"
#include "src/dialect/LetAlgDialect.h"
#include "src/tensor/MetalRuntime.h"
#include "src/tensor/TensorFrontend.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Verifier.h"
#include <stdexcept>

namespace sconeml::tensor {
namespace {
int64_t constantInteger(mlir::Value value) {
  auto constant = value.getDefiningOp<mlir::arith::ConstantOp>();
  auto attr = constant ? mlir::dyn_cast<mlir::IntegerAttr>(constant.getValue())
                       : mlir::IntegerAttr();
  if (!attr)
    throw std::runtime_error("GPU tensor literal requires integer constants");
  return attr.getInt();
}

std::string emitMetalExpression(mlir::Value value,
                                mlir::BlockArgument scalarArgument) {
  if (value == scalarArgument) return "x";
  mlir::Operation *definition = value.getDefiningOp();
  if (!definition)
    throw std::runtime_error("unsupported free value in tensor map body");
  if (mlir::isa<mlir::arith::ConstantOp>(definition))
    return "uint(" + std::to_string(constantInteger(value)) + ")";
  auto binary = [&](const char *symbol) {
    return "(" + emitMetalExpression(definition->getOperand(0), scalarArgument) +
           " " + symbol + " " +
           emitMetalExpression(definition->getOperand(1), scalarArgument) + ")";
  };
  if (mlir::isa<mlir::arith::AddIOp>(definition)) return binary("+");
  if (mlir::isa<mlir::arith::SubIOp>(definition)) return binary("-");
  if (mlir::isa<mlir::arith::MulIOp>(definition)) return binary("*");
  throw std::runtime_error("unsupported operation in Metal tensor map: " +
                           definition->getName().getStringRef().str());
}

std::vector<int32_t> readLiteral(sconeml::letalg::TensorMapOp map) {
  auto type = mlir::dyn_cast<mlir::MemRefType>(map.getInput().getType());
  if (!type || type.getRank() != 1 || !type.hasStaticShape() ||
      !type.getElementType().isInteger(32) ||
      !map.getInput().getDefiningOp<mlir::memref::AllocOp>())
    throw std::runtime_error("GPU runner requires a static rank-1 i32 tensor literal");
  std::vector<int32_t> values(type.getDimSize(0));
  std::vector<bool> initialized(values.size(), false);
  for (mlir::Operation &operation : *map->getBlock()) {
    if (&operation == map.getOperation()) break;
    auto store = mlir::dyn_cast<mlir::memref::StoreOp>(operation);
    if (!store || store.getMemref() != map.getInput()) continue;
    if (store.getIndices().size() != 1)
      throw std::runtime_error("invalid tensor literal index");
    int64_t index = constantInteger(store.getIndices()[0]);
    if (index < 0 || static_cast<size_t>(index) >= values.size() || initialized[index])
      throw std::runtime_error("invalid or repeated tensor literal index");
    values[index] = static_cast<int32_t>(constantInteger(store.getValue()));
    initialized[index] = true;
  }
  for (bool present : initialized)
    if (!present) throw std::runtime_error("tensor literal is not fully initialized");
  return values;
}

std::string translateTensorMapToMetal(sconeml::letalg::TensorMapOp map) {
  mlir::Block &body = map.getBody().front();
  auto yield = mlir::cast<sconeml::letalg::YieldOp>(body.getTerminator());
  std::string expression = emitMetalExpression(yield.getExpr(), body.getArgument(0));
  // Unsigned arithmetic preserves i32 modulo-2^32 behavior on overflow.
  return R"metal(#include <metal_stdlib>
using namespace metal;

kernel void tensor_map(device const uint *input [[buffer(0)]],
                       device uint *output [[buffer(1)]],
                       constant uint &count [[buffer(2)]],
                       uint index [[thread_position_in_grid]]) {
  if (index >= count) return;
  uint x = input[index];
  output[index] = )metal" + expression + R"metal(;
}
)metal";
}
} // namespace

std::vector<int32_t> runTensorLiteralGPU(const std::string &source,
                                         std::string *gpuSource) {
  mlir::MLIRContext context;
  context.getOrLoadDialect<sconeml::letalg::LetAlgDialect>();
  context.getOrLoadDialect<mlir::arith::ArithDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::memref::MemRefDialect>();
  context.getOrLoadDialect<mlir::scf::SCFDialect>();
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));
  translateTensorSource(*module, source);
  if (mlir::failed(mlir::verify(*module)))
    throw std::runtime_error("invalid tensor source IR");
  sconeml::letalg::TensorMapOp map;
  unsigned mapCount = 0;
  module->walk([&](sconeml::letalg::TensorMapOp candidate) {
    map = candidate;
    ++mapCount;
  });
  if (mapCount != 1)
    throw std::runtime_error("GPU runner requires exactly one tensor map");
  auto let = mlir::dyn_cast_or_null<sconeml::letalg::LetOp>(map->getParentOp());
  auto letResult = let ? mlir::dyn_cast<sconeml::letalg::YieldOp>(
                             let.getRegion().front().getTerminator())
                       : sconeml::letalg::YieldOp();
  if (!letResult || letResult.getExpr() != map.getOutput())
    throw std::runtime_error("GPU runner requires the tensor map output as the result");
  auto function = mlir::dyn_cast_or_null<mlir::func::FuncOp>(let->getParentOp());
  auto result = function ? mlir::dyn_cast<sconeml::letalg::YieldOp>(
                               function.getBody().front().getTerminator())
                         : sconeml::letalg::YieldOp();
  if (!result || result.getExpr() != let.getResult())
    throw std::runtime_error("GPU runner requires a top-level tensor literal expression");
  auto input = readLiteral(map);
  std::string metal = translateTensorMapToMetal(map);
  if (gpuSource) *gpuSource = metal;
  std::vector<int32_t> output(input.size());
  runMetalKernel(metal, input.data(), output.data(), input.size());
  return output;
}
bool gpuIsAvailable() { return metalIsAvailable(); }
std::string gpuName() { return metalDeviceName(); }
} // namespace sconeml::tensor
