#include "src/tensor/GPURunner.h"
#include "src/conversion/LowerTensorToGPU.h"
#include "src/conversion/MetalExporter.h"
#include "src/dialect/LetAlgDialect.h"
#include "src/tensor/MetalRuntime.h"
#include "src/tensor/TensorFrontend.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include <stdexcept>
#include <memory>
#include <array>

namespace sconeml::tensor {
namespace {
std::unique_ptr<mlir::MLIRContext> makeGPUContext() {
  auto context = std::make_unique<mlir::MLIRContext>();
  context->getOrLoadDialect<sconeml::letalg::LetAlgDialect>();
  context->getOrLoadDialect<mlir::arith::ArithDialect>();
  context->getOrLoadDialect<mlir::func::FuncDialect>();
  context->getOrLoadDialect<mlir::memref::MemRefDialect>();
  context->getOrLoadDialect<mlir::scf::SCFDialect>();
  return context;
}

sconeml::letalg::TensorMapOp findSingleMap(mlir::ModuleOp module) {
  sconeml::letalg::TensorMapOp map;
  unsigned count = 0;
  module.walk([&](sconeml::letalg::TensorMapOp candidate) {
    map = candidate;
    ++count;
  });
  if (count != 1)
    throw std::runtime_error("GPU runner requires exactly one tensor map");
  return map;
}

std::vector<int32_t> executeMap(sconeml::letalg::TensorMapOp map,
                                 std::span<const MetalInput> inputs,
                                 std::string *gpuSource,
                                 MetalLaunchOptions options) {
  // Check host-representable launch bounds before allocating the host output.
  const auto count = inputs.front().size();
  validateMetalInputs(map.getInputs().size(), inputs, count);
  planMetalLaunch(count, count, count, options);
  auto device = sconeml::lowerTensorMapToGPU(map);
  if (mlir::failed(device))
    throw std::runtime_error("GPU tensor map lowering failed");
  auto kernel = sconeml::exportMetalKernel(**device);
  if (mlir::failed(kernel))
    throw std::runtime_error("GPU Metal export failed");
  if (gpuSource) *gpuSource = kernel->source;
  std::vector<int32_t> output(count);
  runMetalKernel(kernel->source, kernel->entrypoint, kernel->inputCount,
                 inputs, output, count, options);
  return output;
}

int64_t constantInteger(mlir::Value value) {
  auto constant = value.getDefiningOp<mlir::arith::ConstantOp>();
  auto attr = constant ? mlir::dyn_cast<mlir::IntegerAttr>(constant.getValue())
                       : mlir::IntegerAttr();
  if (!attr)
    throw std::runtime_error("GPU tensor literal requires integer constants");
  return attr.getInt();
}

std::vector<int32_t> readLiteral(sconeml::letalg::TensorMapOp map) {
  auto type = mlir::dyn_cast<mlir::MemRefType>(map.getInputs().front().getType());
  if (!type || type.getRank() != 1 || !type.hasStaticShape() ||
      !type.getElementType().isInteger(32) ||
      !map.getInputs().front().getDefiningOp<mlir::memref::AllocOp>())
    throw std::runtime_error("GPU runner requires a static rank-1 i32 tensor literal");
  std::vector<int32_t> values(type.getDimSize(0));
  std::vector<bool> initialized(values.size(), false);
  for (mlir::Operation &operation : *map->getBlock()) {
    if (&operation == map.getOperation()) break;
    auto store = mlir::dyn_cast<mlir::memref::StoreOp>(operation);
    if (!store || store.getMemref() != map.getInputs().front()) continue;
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

} // namespace

std::vector<int32_t> runTensorLiteralGPU(const std::string &source,
                                       std::string *gpuSource,
                                       MetalLaunchOptions options) {
  auto context = makeGPUContext();
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::ModuleOp::create(mlir::UnknownLoc::get(context.get()));
  translateTensorSource(*module, source);
  auto map = findSingleMap(*module);
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
  std::array<MetalInput, 1> inputs{input};
  return executeMap(map, inputs, gpuSource, options);
}

std::vector<int32_t> runTensorGPU(const std::string &expression,
                                 std::span<const TensorArgument> arguments,
                                 std::string *gpuSource,
                                 MetalLaunchOptions options) {
  if (gpuSource)
    gpuSource->clear();
  std::vector<TensorArgumentSpec> specs;
  for (const auto &argument : arguments)
    specs.push_back({argument.name, argument.values.size()});
  auto context = makeGPUContext();
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::ModuleOp::create(mlir::UnknownLoc::get(context.get()));
  translateTensorExpression(*module, expression, specs);
  std::vector<MetalInput> inputs;
  for (const auto &argument : arguments)
    inputs.push_back(argument.values);
  return executeMap(findSingleMap(*module), inputs, gpuSource, options);
}
bool gpuIsAvailable() { return metalIsAvailable(); }
std::string gpuName() { return metalDeviceName(); }
} // namespace sconeml::tensor
