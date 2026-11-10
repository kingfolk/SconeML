#include "src/conversion/LowerShapedTensorToGPU.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Verifier.h"
#include "src/conversion/LowerTensorToGPU.h"
#include "src/conversion/ShapedGPUIndexing.h"
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
  std::vector<std::vector<int64_t>> shapes;
  for (auto input : map.getInputs()) {
    auto shape = mlir::cast<mlir::MemRefType>(input.getType()).getShape();
    shapes.emplace_back(shape.begin(), shape.end());
  }
  auto shape = outputType.getShape();
  auto layout =
      makeBroadcastIndexPlan(shapes, {shape.begin(), shape.end()}).layout;
  if (outputType.getElementType().isF32()) {
    layout.elementType = tensor::MetalElementType::F32;
    layout.floatProfile = tensor::MetalFloatProfile::SignedFTZ;
  }
  return layout;
}

namespace {
mlir::FailureOr<mlir::OwningOpRef<mlir::ModuleOp>>
lowerBroadcast(mlir::linalg::GenericOp generic,
               const std::vector<std::vector<int64_t>> &shapes,
               const std::vector<int64_t> &outputShape,
               const BroadcastIndexPlan &plan) {
  auto *context = generic.getContext();
  context->getOrLoadDialect<mlir::gpu::GPUDialect>();
  context->getOrLoadDialect<mlir::scf::SCFDialect>();
  mlir::OpBuilder builder(context);
  const auto loc = generic.getLoc();
  mlir::OwningOpRef<mlir::ModuleOp> module = mlir::ModuleOp::create(loc);
  builder.setInsertionPointToStart(module->getBody());
  auto device = mlir::gpu::GPUModuleOp::create(builder, loc, "kernels");
  builder.setInsertionPointToStart(device.getBody());
  auto global = mlir::gpu::AddressSpaceAttr::get(
      context, mlir::gpu::AddressSpace::Global);
  auto elementType =
      mlir::cast<mlir::MemRefType>(generic.getOutputs().front().getType())
          .getElementType();
  auto buffer =
      mlir::MemRefType::get({mlir::ShapedType::kDynamic}, elementType,
                            mlir::MemRefLayoutAttrInterface{}, global);
  const auto k = shapes.size();
  llvm::SmallVector<mlir::Type> parameters(k + 1, buffer);
  parameters.push_back(builder.getI32Type());
  auto kernel = mlir::gpu::GPUFuncOp::create(
      builder, loc, "tensor_map", builder.getFunctionType(parameters, {}));
  kernel->setAttr(mlir::gpu::GPUDialect::getKernelFuncAttrName(),
                  builder.getUnitAttr());
  kernel->setAttr("sconeml.broadcast_kernel", builder.getUnitAttr());
  auto &entry = kernel.getBody().front();
  builder.setInsertionPointToStart(&entry);
  auto block =
      mlir::gpu::BlockIdOp::create(builder, loc, mlir::gpu::Dimension::x);
  auto width =
      mlir::gpu::BlockDimOp::create(builder, loc, mlir::gpu::Dimension::x);
  auto thread =
      mlir::gpu::ThreadIdOp::create(builder, loc, mlir::gpu::Dimension::x);
  auto base = mlir::arith::MulIOp::create(builder, loc, block, width);
  auto index = mlir::arith::AddIOp::create(builder, loc, base, thread);
  auto count = mlir::arith::IndexCastUIOp::create(
      builder, loc, builder.getIndexType(), entry.getArgument(k + 1));
  auto inBounds = mlir::arith::CmpIOp::create(
      builder, loc, mlir::arith::CmpIPredicate::ult, index, count);
  mlir::scf::IfOp::create(
      builder, loc, inBounds,
      [&](mlir::OpBuilder &body, mlir::Location bodyLoc) {
        mlir::IRMapping mapping;
        auto &source = generic.getRegion().front();
        for (size_t i = 0; i < k; ++i) {
          mlir::Value inputIndex = index;
          if (plan.inputs[i] == BroadcastIndex::Zero)
            inputIndex = mlir::arith::ConstantIndexOp::create(body, bodyLoc, 0);
          else if (plan.inputs[i] == BroadcastIndex::Row ||
                   plan.inputs[i] == BroadcastIndex::Column) {
            auto columns = mlir::arith::ConstantIndexOp::create(
                body, bodyLoc, outputShape.back());
            if (plan.inputs[i] == BroadcastIndex::Row)
              inputIndex =
                  mlir::arith::DivUIOp::create(body, bodyLoc, index, columns);
            else
              inputIndex =
                  mlir::arith::RemUIOp::create(body, bodyLoc, index, columns);
          }
          auto element =
              mlir::memref::LoadOp::create(body, bodyLoc, entry.getArgument(i),
                                           mlir::ValueRange{inputIndex});
          mapping.map(source.getArgument(i), element.getResult());
        }
        for (auto &op : source.without_terminator())
          body.clone(op, mapping);
        auto yield = mlir::cast<mlir::linalg::YieldOp>(source.getTerminator());
        mlir::memref::StoreOp::create(
            body, bodyLoc, mapping.lookup(yield.getValues().front()),
            entry.getArgument(k), mlir::ValueRange{index});
        mlir::scf::YieldOp::create(body, bodyLoc);
      });
  mlir::gpu::ReturnOp::create(builder, loc);
  llvm::SmallVector<mlir::Attribute> shapeAttrs;
  llvm::SmallVector<int64_t> counts;
  for (const auto &shape : shapes)
    shapeAttrs.push_back(builder.getDenseI64ArrayAttr(shape));
  for (auto count : plan.layout.inputElementCounts)
    counts.push_back(count);
  (*module)->setAttr(
      "sconeml.broadcast",
      builder.getDictionaryAttr(
          {builder.getNamedAttr("shape",
                                builder.getDenseI64ArrayAttr(outputShape)),
           builder.getNamedAttr("input_shapes",
                                builder.getArrayAttr(shapeAttrs)),
           builder.getNamedAttr("input_counts",
                                builder.getDenseI64ArrayAttr(counts)),
           builder.getNamedAttr(
               "output_count",
               builder.getI64IntegerAttr(plan.layout.outputElementCount))}));
  if (elementType.isF32()) {
    (*module)->setAttr("sconeml.element_type",
                       mlir::TypeAttr::get(elementType));
    (*module)->setAttr("sconeml.element_bytes", builder.getI64IntegerAttr(4));
    (*module)->setAttr("sconeml.float_profile",
                       builder.getStringAttr("f32-ftz"));
  }
  if (mlir::failed(mlir::verify(*module)))
    return mlir::failure();
  return std::move(module);
}
} // namespace

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
  std::vector<int64_t> outputShape(shape.begin(), shape.end());
  std::vector<std::vector<int64_t>> inputShapes;
  bool identical = true;
  for (auto input : generic.getInputs()) {
    auto inputShape = mlir::cast<mlir::MemRefType>(input.getType()).getShape();
    inputShapes.emplace_back(inputShape.begin(), inputShape.end());
    identical &= inputShape == shape;
  }
  if (!identical ||
      mlir::cast<mlir::MemRefType>(generic.getOutputs().front().getType())
          .getElementType()
          .isF32())
    return lowerBroadcast(generic, inputShapes, outputShape,
                          makeBroadcastIndexPlan(inputShapes, outputShape));
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
