#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "src/tensor/F32Support.h"
#include "src/tensor/Sum.h"
#include "llvm/Support/TargetSelect.h"
#include <cfenv>
#include <stdexcept>

namespace sconeml::tensor {
namespace {
mlir::Value normalize(mlir::OpBuilder &b, mlir::Location loc, mlir::Value v) {
  // Sum canonicalizes all tiny values/zero to +0; M4 elementwise remains signed
  // FTZ.
  auto bits = mlir::arith::BitcastOp::create(b, loc, b.getI32Type(), v);
  auto mask = mlir::arith::ConstantIntOp::create(b, loc, 0x7fffffff, 32);
  auto magnitude = mlir::arith::AndIOp::create(b, loc, bits, mask);
  auto minimum = mlir::arith::ConstantIntOp::create(b, loc, 0x00800000, 32);
  auto tiny = mlir::arith::CmpIOp::create(
      b, loc, mlir::arith::CmpIPredicate::ult, magnitude, minimum);
  auto zero = mlir::arith::ConstantIntOp::create(b, loc, 0, 32);
  auto selected = mlir::arith::SelectOp::create(b, loc, tiny, zero, bits);
  return mlir::arith::BitcastOp::create(b, loc, b.getF32Type(), selected);
}
} // namespace
void buildSumCPU(mlir::ModuleOp m, const SumPlan &p, bool f32) {
  validateSumPlan(p);
  if (!m.getBody()->empty())
    throw std::invalid_argument(
        "sum builder requires an empty module; raw IR is unsupported");
  auto *ctx = m.getContext();
  ctx->getOrLoadDialect<mlir::linalg::LinalgDialect>();
  ctx->getOrLoadDialect<mlir::scf::SCFDialect>();
  ctx->getOrLoadDialect<mlir::memref::MemRefDialect>();
  ctx->getOrLoadDialect<mlir::func::FuncDialect>();
  ctx->getOrLoadDialect<mlir::arith::ArithDialect>();
  mlir::OpBuilder b(ctx);
  auto loc = b.getUnknownLoc();
  b.setInsertionPointToStart(m.getBody());
  mlir::Type element =
      f32 ? mlir::Type(b.getF32Type()) : mlir::Type(b.getI32Type());
  auto inType =
      mlir::MemRefType::get({static_cast<int64_t>(p.inputCount)}, element);
  auto outType =
      mlir::MemRefType::get({static_cast<int64_t>(p.outputCount)}, element);
  auto fn = mlir::func::FuncOp::create(
      b, loc, "sum_tree", b.getFunctionType({inType, outType}, {}));
  fn->setAttr("llvm.emit_c_interface", b.getUnitAttr());
  fn->setAttr("sconeml.sum_tree_width", b.getI64IntegerAttr(256));
  if (f32)
    fn->setAttr("sconeml.float_profile",
                b.getStringAttr("f32-sum-tree256-ftz"));
  auto *entry = fn.addEntryBlock();
  b.setInsertionPointToStart(entry);
  auto zero = mlir::arith::ConstantOp::create(b, loc, b.getZeroAttr(element));
  mlir::linalg::FillOp::create(b, loc, mlir::ValueRange{zero},
                               mlir::ValueRange{entry->getArgument(1)});
  if (p.outputCount && p.length) {
    auto c = [&](int64_t n) {
      return mlir::arith::ConstantIndexOp::create(b, loc, n).getResult();
    };
    auto low = c(0), one = c(1), rows = c(p.outputCount);
    mlir::Value input = entry->getArgument(0);
    size_t length = p.length, baseStride = p.baseStride,
           inputStride = p.inputStride;
    bool ownsInput = false;
    do {
      auto padded = ((length + 255) / 256) * 256;
      auto flat = mlir::MemRefType::get(
          {static_cast<int64_t>(p.outputCount * padded)}, element);
      auto buffer = mlir::memref::AllocOp::create(b, loc, flat);
      mlir::linalg::FillOp::create(b, loc, mlir::ValueRange{zero},
                                   mlir::ValueRange{buffer});
      auto bound = c(length), stride = c(inputStride), base = c(baseStride),
           width = c(padded);
      mlir::scf::ForOp::create(
          b, loc, low, rows, one, mlir::ValueRange{},
          [&](mlir::OpBuilder &rb, mlir::Location l, mlir::Value row,
              mlir::ValueRange) {
            mlir::scf::ForOp::create(
                rb, l, low, bound, one, mlir::ValueRange{},
                [&](mlir::OpBuilder &kb, mlir::Location kl, mlir::Value k,
                    mlir::ValueRange) {
                  auto r = mlir::arith::MulIOp::create(kb, kl, row, base);
                  auto j = mlir::arith::MulIOp::create(kb, kl, k, stride);
                  auto index = mlir::arith::AddIOp::create(kb, kl, r, j);
                  mlir::Value v = mlir::memref::LoadOp::create(
                      kb, kl, input, mlir::ValueRange{index});
                  if (f32)
                    v = normalize(kb, kl, v);
                  auto destBase =
                      mlir::arith::MulIOp::create(kb, kl, row, width);
                  auto dest = mlir::arith::AddIOp::create(kb, kl, destBase, k);
                  mlir::memref::StoreOp::create(kb, kl, v, buffer,
                                                mlir::ValueRange{dest});
                  mlir::scf::YieldOp::create(kb, kl);
                });
            mlir::scf::YieldOp::create(rb, l);
          });
      if (ownsInput)
        mlir::memref::DeallocOp::create(b, loc, input);
      mlir::Value current = buffer;
      size_t n = padded;
      for (unsigned depth = 0; depth < 8; ++depth) {
        auto pairs = n / 2;
        std::vector<int64_t> shape{static_cast<int64_t>(p.outputCount),
                                   static_cast<int64_t>(pairs), 2};
        std::vector<int64_t> strides{static_cast<int64_t>(n), 2, 1};
        auto viewType = mlir::MemRefType::get(
            shape, element, mlir::StridedLayoutAttr::get(ctx, 0, strides));
        auto view = mlir::memref::ReinterpretCastOp::create(
            b, loc, viewType, current, 0, shape, strides);
        auto next = mlir::memref::AllocOp::create(
            b, loc,
            mlir::MemRefType::get({static_cast<int64_t>(p.outputCount * pairs)},
                                  element));
        std::vector<int64_t> nextShape{static_cast<int64_t>(p.outputCount),
                                       static_cast<int64_t>(pairs)},
            nextStrides{static_cast<int64_t>(pairs), 1};
        auto nextType = mlir::MemRefType::get(
            nextShape, element,
            mlir::StridedLayoutAttr::get(ctx, 0, nextStrides));
        auto nextView = mlir::memref::ReinterpretCastOp::create(
            b, loc, nextType, next, 0, nextShape, nextStrides);
        mlir::linalg::FillOp::create(b, loc, mlir::ValueRange{zero},
                                     mlir::ValueRange{next});
        auto outputMap = mlir::AffineMap::get(
            3, 0, {b.getAffineDimExpr(0), b.getAffineDimExpr(1)}, ctx);
        mlir::linalg::GenericOp::create(
            b, loc, mlir::ValueRange{view}, mlir::ValueRange{nextView},
            llvm::ArrayRef<mlir::AffineMap>{b.getMultiDimIdentityMap(3),
                                            outputMap},
            llvm::ArrayRef<mlir::utils::IteratorType>{
                mlir::utils::IteratorType::parallel,
                mlir::utils::IteratorType::parallel,
                mlir::utils::IteratorType::reduction},
            [&](mlir::OpBuilder &body, mlir::Location l,
                mlir::ValueRange values) {
              mlir::Value sum = f32 ? mlir::Value(mlir::arith::AddFOp::create(
                                          body, l, values[1], values[0]))
                                    : mlir::Value(mlir::arith::AddIOp::create(
                                          body, l, values[1], values[0]));
              if (f32)
                sum = normalize(body, l, sum);
              mlir::linalg::YieldOp::create(body, l, sum);
            });
        mlir::memref::DeallocOp::create(b, loc, current);
        current = next;
        n = pairs;
      }
      input = current;
      ownsInput = true;
      length = n;
      baseStride = n;
      inputStride = 1;
    } while (length > 1);
    mlir::scf::ForOp::create(
        b, loc, low, rows, one, mlir::ValueRange{},
        [&](mlir::OpBuilder &body, mlir::Location l, mlir::Value row,
            mlir::ValueRange) {
          auto v = mlir::memref::LoadOp::create(body, l, input,
                                                mlir::ValueRange{row});
          mlir::memref::StoreOp::create(body, l, v, entry->getArgument(1),
                                        mlir::ValueRange{row});
          mlir::scf::YieldOp::create(body, l);
        });
    mlir::memref::DeallocOp::create(b, loc, input);
  }
  mlir::func::ReturnOp::create(b, loc);
  if (mlir::failed(mlir::verify(m)))
    throw std::runtime_error("sum CPU IR verification failed");
}
namespace {
template <typename T, typename Arg>
SumResult<T> run(const std::string &source, std::span<const Arg> args,
                 std::string *dump) {
  if (dump)
    dump->clear();
  constexpr bool f32 = std::is_same_v<T, float>;
  if (f32 && std::fegetround() != FE_TONEAREST)
    throw std::invalid_argument(
        "sum requires round-to-nearest host environment");
  std::vector<ShapedTensorArgumentSpec> specs;
  for (const auto &a : args) {
    if (!a.values.empty() && !a.values.data())
      throw std::invalid_argument(
          "sum nonempty input requires nonnull storage");
    specs.push_back({a.name, a.values.size(), a.shape});
  }
  auto plan = prepareSum(source, specs, f32);
  SumResult<T> result{std::vector<T>(plan.outputCount), plan.outputShape};
  if (!plan.outputCount || !plan.length)
    return result;
  static bool initialized = [] {
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    return true;
  }();
  (void)initialized;
  mlir::DialectRegistry registry;
  mlir::registerAllToLLVMIRTranslations(registry);
  mlir::MLIRContext context(registry);
  auto module = mlir::OwningOpRef<mlir::ModuleOp>(
      mlir::ModuleOp::create(mlir::UnknownLoc::get(&context)));
  buildSumCPU(*module, plan, f32);
  mlir::PassManager pm(&context);
  pm.addNestedPass<mlir::func::FuncOp>(mlir::createConvertLinalgToLoopsPass());
  pm.addPass(mlir::createLowerAffinePass());
  pm.addPass(mlir::createSCFToControlFlowPass());
  pm.addPass(mlir::createConvertControlFlowToLLVMPass());
  pm.addPass(mlir::createArithToLLVMConversionPass());
  pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
  pm.addPass(mlir::createConvertFuncToLLVMPass());
  pm.addPass(mlir::createReconcileUnrealizedCastsPass());
  if (mlir::failed(pm.run(*module)))
    throw std::runtime_error("sum CPU LLVM lowering failed");
  if (dump) {
    llvm::raw_string_ostream os(*dump);
    module->print(os);
  }
  bool llvmOnly = true;
  module->walk([&](mlir::Operation *op) {
    if (op != module->getOperation() &&
        op->getDialect()->getNamespace() != "llvm")
      llvmOnly = false;
  });
  if (!llvmOnly)
    throw std::runtime_error("sum lowering left non-LLVM operations");
  mlir::ExecutionEngineOptions options;
  auto transform = mlir::makeOptimizingTransformer(2, 0, nullptr);
  options.transformer = transform;
  auto engine = mlir::ExecutionEngine::create(*module, options);
  if (!engine)
    throw std::runtime_error(llvm::toString(engine.takeError()));
  auto function = (*engine)->lookup("_mlir_ciface_sum_tree");
  if (!function)
    throw std::runtime_error(llvm::toString(function.takeError()));
  auto *data = const_cast<T *>(args.front().values.data());
  StridedMemRefType<T, 1> in{
      data, data, 0, {static_cast<int64_t>(plan.inputCount)}, {1}},
      out{result.values.data(),
          result.values.data(),
          0,
          {static_cast<int64_t>(plan.outputCount)},
          {1}};
  reinterpret_cast<void (*)(StridedMemRefType<T, 1> *,
                            StridedMemRefType<T, 1> *)>(*function)(&in, &out);
  return result;
}
} // namespace
SumI32Result runSumI32CPU(const std::string &s,
                          std::span<const ShapedTensorArgument> a,
                          std::string *ir) {
  return run<int32_t>(s, a, ir);
}
SumF32Result runSumF32CPU(const std::string &s,
                          std::span<const ShapedTensorF32Argument> a,
                          std::string *ir) {
  return run<float>(s, a, ir);
}
} // namespace sconeml::tensor
