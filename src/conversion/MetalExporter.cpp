#include "src/conversion/MetalExporter.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/raw_ostream.h"

#include <iterator>

namespace sconeml {
namespace {

void reject(mlir::Operation *op, llvm::StringRef reason) {
  op->emitOpError("MSL export: ") << reason;
}

bool isGlobalBuffer(mlir::Type type) {
  auto buffer = mlir::dyn_cast<mlir::MemRefType>(type);
  if (!buffer || buffer.getRank() != 1 || !buffer.isDynamicDim(0) ||
      !buffer.getElementType().isSignlessInteger(32) ||
      !buffer.getLayout().isIdentity())
    return false;
  auto space = mlir::dyn_cast_or_null<mlir::gpu::AddressSpaceAttr>(
      buffer.getMemorySpace());
  return space && space.getValue() == mlir::gpu::AddressSpace::Global;
}

bool supportedOperation(mlir::Operation *op) {
  return mlir::isa<
      mlir::ModuleOp, mlir::gpu::GPUModuleOp, mlir::gpu::GPUFuncOp,
      mlir::gpu::BlockIdOp, mlir::gpu::BlockDimOp, mlir::gpu::ThreadIdOp,
      mlir::gpu::ReturnOp, mlir::arith::ConstantOp, mlir::arith::AddIOp,
      mlir::arith::SubIOp, mlir::arith::MulIOp, mlir::arith::IndexCastUIOp,
      mlir::arith::CmpIOp, mlir::scf::IfOp, mlir::scf::YieldOp,
      mlir::memref::LoadOp, mlir::memref::StoreOp>(op);
}

// Validate the complete envelope before emitting any source. This is a
// restricted exporter, not a general lowering of arbitrary MLIR index values.
mlir::FailureOr<mlir::gpu::GPUFuncOp> validate(mlir::ModuleOp module) {
  if (mlir::failed(mlir::verify(module))) {
    reject(module, "input IR verification failed");
    return mlir::failure();
  }
  auto walk = module.walk([](mlir::Operation *op) {
    if (!supportedOperation(op)) {
      reject(op, "unsupported operation");
      return mlir::WalkResult::interrupt();
    }
    return mlir::WalkResult::advance();
  });
  if (walk.wasInterrupted())
    return mlir::failure();
  auto devices = module.getOps<mlir::gpu::GPUModuleOp>();
  if (!llvm::hasSingleElement(module.getBody()->getOperations()) ||
      !llvm::hasSingleElement(devices)) {
    reject(module, "expected exactly one gpu.module");
    return mlir::failure();
  }
  auto device = *devices.begin();
  auto kernels = device.getOps<mlir::gpu::GPUFuncOp>();
  if (!llvm::hasSingleElement(device.getBody()->getOperations()) ||
      !llvm::hasSingleElement(kernels)) {
    reject(device, "expected exactly one defined GPU kernel");
    return mlir::failure();
  }
  auto kernel = *kernels.begin();
  auto type = kernel.getFunctionType();
  if (!kernel.isKernel() || !llvm::hasSingleElement(kernel.getBody()) ||
      type.getNumInputs() != 3 || type.getNumResults() != 0 ||
      !isGlobalBuffer(type.getInput(0)) || !isGlobalBuffer(type.getInput(1)) ||
      !type.getInput(2).isSignlessInteger(32) ||
      kernel.getBody().front().getNumArguments() != 3) {
    reject(kernel,
           "expected global identity-layout rank-1 dynamic i32 "
           "input/output and an i32 count, with no memory attributions");
    return mlir::failure();
  }
  auto &entry = kernel.getBody().front();
  llvm::SmallVector<mlir::Operation *> ops;
  for (auto &op : entry)
    ops.push_back(&op);
  const llvm::StringRef expected[] = {
      "gpu.block_id", "gpu.block_dim", "gpu.thread_id",
      "arith.muli",   "arith.addi",    "arith.index_castui",
      "arith.cmpi",   "scf.if",        "gpu.return"};
  if (ops.size() != std::size(expected)) {
    reject(kernel,
           "expected the one-dimensional map index/guard/return envelope");
    return mlir::failure();
  }
  for (size_t i = 0; i < ops.size(); ++i)
    if (ops[i]->getName().getStringRef() != expected[i]) {
      reject(ops[i], "unexpected operation in the map kernel envelope");
      return mlir::failure();
    }
  auto block = mlir::cast<mlir::gpu::BlockIdOp>(ops[0]);
  auto width = mlir::cast<mlir::gpu::BlockDimOp>(ops[1]);
  auto thread = mlir::cast<mlir::gpu::ThreadIdOp>(ops[2]);
  auto base = mlir::cast<mlir::arith::MulIOp>(ops[3]);
  auto index = mlir::cast<mlir::arith::AddIOp>(ops[4]);
  auto count = mlir::cast<mlir::arith::IndexCastUIOp>(ops[5]);
  auto cmp = mlir::cast<mlir::arith::CmpIOp>(ops[6]);
  auto guard = mlir::cast<mlir::scf::IfOp>(ops[7]);
  if (block.getDimension() != mlir::gpu::Dimension::x ||
      width.getDimension() != mlir::gpu::Dimension::x ||
      thread.getDimension() != mlir::gpu::Dimension::x ||
      base.getLhs() != block.getResult() ||
      base.getRhs() != width.getResult() ||
      index.getLhs() != base.getResult() ||
      index.getRhs() != thread.getResult() ||
      count.getIn() != entry.getArgument(2) || !count.getType().isIndex() ||
      cmp.getPredicate() != mlir::arith::CmpIPredicate::ult ||
      cmp.getLhs() != index.getResult() || cmp.getRhs() != count.getResult() ||
      guard.getCondition() != cmp.getResult() || guard.getNumResults() != 0 ||
      !guard.getElseRegion().empty() ||
      !llvm::hasSingleElement(guard.getThenRegion())) {
    reject(kernel,
           "expected x-dimensional linear indexing, unsigned count "
           "cast/comparison, and a result-free bounds guard without else");
    return mlir::failure();
  }
  for (auto *op : {ops[3], ops[4]}) {
    auto flags =
        mlir::cast<mlir::arith::ArithIntegerOverflowFlagsInterface>(op);
    if (flags.hasNoSignedWrap() || flags.hasNoUnsignedWrap()) {
      reject(op, "no-wrap promises are not supported");
      return mlir::failure();
    }
  }

  auto &body = guard.getThenRegion().front();
  if (body.getOperations().size() < 3) {
    reject(guard,
           "expected a guarded load, scalar computation, store and yield");
    return mlir::failure();
  }
  auto load = mlir::dyn_cast<mlir::memref::LoadOp>(&body.front());
  auto store =
      mlir::dyn_cast<mlir::memref::StoreOp>(&*std::prev(body.end(), 2));
  auto yield = mlir::dyn_cast<mlir::scf::YieldOp>(&body.back());
  if (!load || !store || !yield || yield.getNumOperands() != 0 ||
      load.getMemref() != entry.getArgument(0) ||
      store.getMemref() != entry.getArgument(1) ||
      load.getIndices().size() != 1 || store.getIndices().size() != 1 ||
      load.getIndices()[0] != index.getResult() ||
      store.getIndices()[0] != index.getResult()) {
    reject(guard,
           "expected input[index] load and output[index] store inside guard");
    return mlir::failure();
  }
  llvm::DenseSet<mlir::Value> scalars{load.getResult()};
  for (auto it = std::next(body.begin()); &*it != store.getOperation(); ++it) {
    auto *op = &*it;
    if (!mlir::isa<mlir::arith::ConstantOp, mlir::arith::AddIOp,
                   mlir::arith::SubIOp, mlir::arith::MulIOp>(op) ||
        op->getNumResults() != 1 ||
        !op->getResult(0).getType().isSignlessInteger(32)) {
      reject(op, "scalar body supports only i32 constants and add/sub/mul");
      return mlir::failure();
    }
    for (auto operand : op->getOperands())
      if (!scalars.contains(operand)) {
        reject(op, "scalar operand is not a local element value");
        return mlir::failure();
      }
    if (auto constant = mlir::dyn_cast<mlir::arith::ConstantOp>(op)) {
      if (!mlir::isa<mlir::IntegerAttr>(constant.getValue())) {
        reject(op, "expected an i32 integer constant");
        return mlir::failure();
      }
    } else {
      auto flags =
          mlir::cast<mlir::arith::ArithIntegerOverflowFlagsInterface>(op);
      if (flags.hasNoSignedWrap() || flags.hasNoUnsignedWrap()) {
        reject(op, "no-wrap promises are not supported");
        return mlir::failure();
      }
    }
    scalars.insert(op->getResult(0));
  }
  if (!scalars.contains(store.getValue())) {
    reject(store, "stored value must be a local i32 element result");
    return mlir::failure();
  }
  return kernel;
}

class Emitter {
public:
  explicit Emitter(llvm::raw_ostream &stream) : stream(stream) {}

  void emitBlock(mlir::Block &block, unsigned indent) {
    for (auto &op : block) {
      if (auto guard = mlir::dyn_cast<mlir::scf::IfOp>(op)) {
        stream.indent(indent)
            << "if (" << names.lookup(guard.getCondition()) << ") {\n";
        emitBlock(guard.getThenRegion().front(), indent + 2);
        stream.indent(indent) << "}\n";
      } else if (auto store = mlir::dyn_cast<mlir::memref::StoreOp>(op)) {
        stream.indent(indent)
            << "output[" << names.lookup(store.getIndices()[0])
            << "] = " << names.lookup(store.getValue()) << ";\n";
      } else if (mlir::isa<mlir::gpu::ReturnOp, mlir::scf::YieldOp>(op)) {
        // Function/block delimiters represent these zero-value terminators.
      } else {
        auto name = "v" + std::to_string(nextValue++);
        names[op.getResult(0)] = name;
        stream.indent(indent)
            << "const "
            << (op.getResult(0).getType().isInteger(1) ? "bool " : "uint ")
            << name << " = ";
        if (mlir::isa<mlir::gpu::BlockIdOp>(op))
          stream << "group_id.x";
        else if (mlir::isa<mlir::gpu::BlockDimOp>(op))
          stream << "group_size.x";
        else if (mlir::isa<mlir::gpu::ThreadIdOp>(op))
          stream << "local_id.x";
        else if (auto constant = mlir::dyn_cast<mlir::arith::ConstantOp>(op))
          stream << mlir::cast<mlir::IntegerAttr>(constant.getValue())
                        .getValue()
                        .getZExtValue()
                 << "u";
        else if (auto load = mlir::dyn_cast<mlir::memref::LoadOp>(op))
          stream << "input[" << names.lookup(load.getIndices()[0]) << "]";
        else if (mlir::isa<mlir::arith::IndexCastUIOp>(op))
          stream << names.lookup(op.getOperand(0));
        else {
          const char *symbol = mlir::isa<mlir::arith::AddIOp>(op)   ? " + "
                               : mlir::isa<mlir::arith::SubIOp>(op) ? " - "
                               : mlir::isa<mlir::arith::MulIOp>(op) ? " * "
                                                                    : " < ";
          stream << names.lookup(op.getOperand(0)) << symbol
                 << names.lookup(op.getOperand(1));
        }
        stream << ";\n";
      }
    }
  }

  llvm::DenseMap<mlir::Value, std::string> names;

private:
  llvm::raw_ostream &stream;
  unsigned nextValue = 0;
};

} // namespace

mlir::FailureOr<MetalKernelSource> exportMetalKernel(mlir::ModuleOp module) {
  auto kernel = validate(module);
  if (mlir::failed(kernel))
    return mlir::failure();
  MetalKernelSource result;
  result.entrypoint = "tensor_map";
  llvm::raw_string_ostream stream(result.source);
  stream << "#include <metal_stdlib>\nusing namespace metal;\n\n"
            "kernel void tensor_map(device const uint *input [[buffer(0)]],\n"
            "                       device uint *output [[buffer(1)]],\n"
            "                       constant uint &count [[buffer(2)]],\n"
            "                       uint3 group_id "
            "[[threadgroup_position_in_grid]],\n"
            "                       uint3 group_size "
            "[[threads_per_threadgroup]],\n"
            "                       uint3 local_id "
            "[[thread_position_in_threadgroup]]) {\n";
  Emitter emitter(stream);
  auto &entry = (*kernel).getBody().front();
  emitter.names[entry.getArgument(0)] = "input";
  emitter.names[entry.getArgument(1)] = "output";
  emitter.names[entry.getArgument(2)] = "count";
  emitter.emitBlock(entry, 2);
  stream << "}\n";
  return result;
}

} // namespace sconeml
