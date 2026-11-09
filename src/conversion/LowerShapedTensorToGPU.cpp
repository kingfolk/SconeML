#include "src/conversion/LowerShapedTensorToGPU.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Verifier.h"
#include "src/conversion/LowerTensorToGPU.h"
#include "src/tensor/TensorFrontend.h"
#include <limits>
#include <stdexcept>

namespace sconeml {
tensor::MetalKernelLayout validateShapedTensorGPU(mlir::ModuleOp module) {
  tensor::validateShapedTensorExpression(module);
  auto function = *module.getOps<mlir::func::FuncOp>().begin();
  auto map = *function.getOps<mlir::linalg::GenericOp>().begin();
  auto outputType =
      mlir::cast<mlir::MemRefType>(map.getOutputs().front().getType());
  for (auto input : map.getInputs())
    if (mlir::cast<mlir::MemRefType>(input.getType()).getShape() !=
        outputType.getShape())
      throw std::invalid_argument("shaped GPU broadcasting is not supported "
                                  "yet; input shapes must match");
  const auto count = static_cast<size_t>(outputType.getNumElements());
  if (count > std::numeric_limits<uint32_t>::max())
    throw std::invalid_argument("shaped GPU element count exceeds u32 range");
  return {std::vector<size_t>(map.getInputs().size(), count), count};
}

mlir::FailureOr<mlir::OwningOpRef<mlir::ModuleOp>>
lowerShapedTensorToGPU(mlir::ModuleOp module) {
  tensor::MetalKernelLayout layout;
  try {
    layout = validateShapedTensorGPU(module);
  } catch (const std::exception &error) {
    module.emitError(error.what());
    return mlir::failure();
  }
  auto function = *module.getOps<mlir::func::FuncOp>().begin();
  auto generic = *function.getOps<mlir::linalg::GenericOp>().begin();
  auto shape =
      mlir::cast<mlir::MemRefType>(generic.getOutputs().front().getType())
          .getShape();
  auto *context = module.getContext();
  context->getOrLoadDialect<letalg::LetAlgDialect>();
  mlir::OpBuilder builder(context);
  const auto loc = generic.getLoc();
  mlir::OwningOpRef<mlir::ModuleOp> flat = mlir::ModuleOp::create(loc);
  builder.setInsertionPointToStart(flat->getBody());
  auto buffer = mlir::MemRefType::get(
      {static_cast<int64_t>(layout.outputElementCount)}, builder.getI32Type());
  const auto k = layout.inputElementCounts.size();
  auto wrapper = mlir::func::FuncOp::create(
      builder, loc, "shaped_flat_map",
      builder.getFunctionType(llvm::SmallVector<mlir::Type>(k + 1, buffer),
                              {}));
  auto *entry = wrapper.addEntryBlock();
  builder.setInsertionPointToStart(entry);
  auto map = letalg::TensorMapOp::create(builder, loc,
                                         entry->getArguments().drop_back(),
                                         entry->getArguments().back());
  llvm::SmallVector<mlir::Type> types(k, builder.getI32Type());
  llvm::SmallVector<mlir::Location> locations(k, loc);
  auto *body = builder.createBlock(&map.getBody(), {}, types, locations);
  builder.setInsertionPointToStart(body);
  mlir::IRMapping mapping;
  auto &source = generic.getRegion().front();
  for (size_t i = 0; i < k; ++i)
    mapping.map(source.getArgument(i), body->getArgument(i));
  for (auto &op : source.without_terminator())
    builder.clone(op, mapping);
  auto yield = mlir::cast<mlir::linalg::YieldOp>(source.getTerminator());
  letalg::YieldOp::create(builder, loc,
                          mapping.lookup(yield.getValues().front()));
  builder.setInsertionPointAfter(map);
  mlir::func::ReturnOp::create(builder, loc);
  if (mlir::failed(mlir::verify(*flat)))
    return mlir::failure();
  auto device = lowerTensorMapToGPU(map);
  if (mlir::failed(device))
    return mlir::failure();
  (**device).walk([&](mlir::gpu::GPUFuncOp kernel) {
    kernel->setAttr("sconeml.shaped_kernel", builder.getUnitAttr());
  });
  llvm::SmallVector<int64_t> counts(k, layout.outputElementCount);
  (**device)->setAttr(
      "sconeml.shaped",
      builder.getDictionaryAttr(
          {builder.getNamedAttr("shape", builder.getDenseI64ArrayAttr(shape)),
           builder.getNamedAttr("input_counts",
                                builder.getDenseI64ArrayAttr(counts)),
           builder.getNamedAttr(
               "output_count",
               builder.getI64IntegerAttr(layout.outputElementCount))}));
  return std::move(*device);
}
} // namespace sconeml
