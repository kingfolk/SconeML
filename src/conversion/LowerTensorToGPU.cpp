#include "src/conversion/LowerTensorToGPU.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Verifier.h"

#include <utility>

namespace sconeml {

mlir::FailureOr<mlir::OwningOpRef<mlir::ModuleOp>>
lowerTensorMapToGPU(letalg::TensorMapOp map) {
  // Verify before accessing the region or cloning its values. In particular,
  // no host capture or unsupported scalar operation may enter device IR.
  if (mlir::failed(mlir::verify(map)))
    return mlir::failure();
  auto inputType = mlir::cast<mlir::MemRefType>(map.getInputs().front().getType());
  if (!inputType.getLayout().isIdentity() || inputType.getMemorySpace()) {
    map.emitOpError("GPU lowering requires identity layout in the default host "
                    "memory space");
    return mlir::failure();
  }

  auto *context = map.getContext();
  context->getOrLoadDialect<mlir::gpu::GPUDialect>();
  context->getOrLoadDialect<mlir::arith::ArithDialect>();
  context->getOrLoadDialect<mlir::memref::MemRefDialect>();
  context->getOrLoadDialect<mlir::scf::SCFDialect>();
  mlir::OpBuilder builder(context);
  auto loc = map.getLoc();
  mlir::OwningOpRef<mlir::ModuleOp> module = mlir::ModuleOp::create(loc);
  builder.setInsertionPointToStart(module->getBody());
  auto device = mlir::gpu::GPUModuleOp::create(builder, loc, "kernels");
  builder.setInsertionPointToStart(device.getBody());
  auto global = mlir::gpu::AddressSpaceAttr::get(
      context, mlir::gpu::AddressSpace::Global);
  auto bufferType =
      mlir::MemRefType::get({mlir::ShapedType::kDynamic}, builder.getI32Type(),
                            mlir::MemRefLayoutAttrInterface{}, global);
  const auto inputCount = map.getInputs().size();
  llvm::SmallVector<mlir::Type> parameters(inputCount + 1, bufferType);
  parameters.push_back(builder.getI32Type());
  auto kernel = mlir::gpu::GPUFuncOp::create(
      builder, loc, "tensor_map",
      builder.getFunctionType(parameters, {}));
  kernel->setAttr(mlir::gpu::GPUDialect::getKernelFuncAttrName(),
                  builder.getUnitAttr());
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
      builder, loc, builder.getIndexType(), entry.getArgument(inputCount + 1));
  auto inBounds = mlir::arith::CmpIOp::create(
      builder, loc, mlir::arith::CmpIPredicate::ult, index, count);
  auto &body = map.getBody().front();
  auto yield = mlir::cast<letalg::YieldOp>(body.getTerminator());
  mlir::scf::IfOp::create(
      builder, loc, inBounds,
      [&](mlir::OpBuilder &bodyBuilder, mlir::Location bodyLoc) {
        mlir::IRMapping mapping;
        for (unsigned i = 0; i < inputCount; ++i) {
          auto element = mlir::memref::LoadOp::create(
              bodyBuilder, bodyLoc, entry.getArgument(i), mlir::ValueRange{index},
              mlir::BoolAttr{}, mlir::IntegerAttr{});
          mapping.map(body.getArgument(i), element.getResult());
        }
        for (mlir::Operation &operation : body.without_terminator())
          bodyBuilder.clone(operation, mapping);
        mlir::memref::StoreOp::create(
            bodyBuilder, bodyLoc, mapping.lookup(yield.getExpr()),
            entry.getArgument(inputCount), mlir::ValueRange{index}, mlir::BoolAttr{},
            mlir::IntegerAttr{});
        mlir::scf::YieldOp::create(bodyBuilder, bodyLoc);
      });
  mlir::gpu::ReturnOp::create(builder, loc);
  if (mlir::failed(mlir::verify(*module)))
    return mlir::failure();
  return std::move(module);
}

} // namespace sconeml
