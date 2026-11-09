#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Diagnostics.h"
#include "src/conversion/LowerShapedTensorToGPU.h"
#include "src/conversion/MetalExporter.h"
#include "src/tensor/GPURunner.h"
#include "src/tensor/TensorFrontend.h"
#include "test/runner/TestSpec.h"
#include "llvm/Support/raw_ostream.h"
#include <array>
#include <limits>

using namespace sconeml::tensor;
using sconeml::test::require;
namespace {
std::vector<ShapedTensorArgument>
arguments(const sconeml::test::TestCase &test) {
  std::vector<ShapedTensorArgument> result;
  for (const auto &arg : *test.runs.front().arguments)
    result.push_back({arg.name, arg.values, *arg.shape});
  return result;
}
std::vector<int32_t> expected(const sconeml::test::TestCase &test) {
  for (const auto &assertion : test.assertions.at("cpu"))
    if (assertion.kind == "result")
      return sconeml::test::parseNumber(assertion.value).values;
  throw std::runtime_error("missing oracle expectation");
}
std::string print(mlir::ModuleOp module) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  module.print(stream);
  return text;
}
template <typename F> void rejects(const char *reason, F action) {
  try {
    action();
  } catch (const std::exception &error) {
    require(std::string(error.what()).find(reason) != std::string::npos,
            std::string("wrong broadcast rejection: ") + error.what());
    return;
  }
  throw std::runtime_error(std::string("accepted invalid broadcast: ") +
                           reason);
}
} // namespace

void checkBroadcastGPUContracts(const std::string &path) {
  auto test = sconeml::test::readCase(path, "matrix-row-column");
  auto args = arguments(test);
  sconeml::MetalKernelSource owned;
  {
    mlir::MLIRContext context;
    context.getOrLoadDialect<mlir::arith::ArithDialect>();
    context.getOrLoadDialect<mlir::func::FuncDialect>();
    context.getOrLoadDialect<mlir::memref::MemRefDialect>();
    mlir::ScopedDiagnosticHandler quiet(
        &context, [](mlir::Diagnostic &) { return mlir::success(); });
    mlir::OwningOpRef<mlir::ModuleOp> host =
        mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));
    std::vector<ShapedTensorArgumentSpec> specs;
    for (const auto &arg : args)
      specs.push_back({arg.name, arg.values.size(), arg.shape});
    translateShapedTensorExpression(*host, test.program, specs);
    const auto before = print(*host);
    auto device = sconeml::lowerShapedTensorToGPU(*host);
    require(mlir::succeeded(device) && print(*host) == before,
            "broadcast extraction modified host module");
    const auto deviceBefore = print(**device);
    auto exported = sconeml::exportMetalKernel(**device);
    require(mlir::succeeded(exported) && print(**device) == deviceBefore &&
                exported->shapedLayout &&
                exported->shapedLayout->inputElementCounts ==
                    std::vector<size_t>({6, 3, 2}) &&
                exported->shapedLayout->outputElementCount == 6,
            "broadcast export counts/preservation failed");
    owned = *exported;
    auto corrupt = [&](auto mutate) {
      mlir::OwningOpRef<mlir::ModuleOp> bad = (**device).clone();
      mlir::OpBuilder builder(&context);
      mutate(*bad, builder);
      const auto original = print(*bad);
      require(mlir::failed(sconeml::exportMetalKernel(*bad)) &&
                  print(*bad) == original,
              "exporter accepted or modified malformed broadcast IR");
    };
    corrupt(
        [](auto module, auto &) { module->removeAttr("sconeml.broadcast"); });
    corrupt([](auto module, auto &) {
      module.walk([](mlir::gpu::GPUFuncOp kernel) {
        kernel->removeAttr("sconeml.broadcast_kernel");
      });
    });
    // Removing both tags must not broadly authorize div/rem in legacy kernels.
    corrupt([](auto module, auto &) {
      module->removeAttr("sconeml.broadcast");
      module.walk([](mlir::gpu::GPUFuncOp kernel) {
        kernel->removeAttr("sconeml.broadcast_kernel");
      });
    });
    for (int mode = 0; mode < 8; ++mode)
      corrupt([&](auto module, auto &b) {
        auto metadata = module->template getAttrOfType<mlir::DictionaryAttr>(
            "sconeml.broadcast");
        mlir::NamedAttrList fields(metadata);
        if (mode == 0)
          fields.set("input_counts", b.getDenseI64ArrayAttr({6, 4, 2}));
        if (mode == 1)
          fields.set("shape", b.getDenseI64ArrayAttr({3, 2}));
        if (mode == 2)
          fields.set("output_count", b.getI64IntegerAttr(7));
        if (mode == 3)
          fields.set("input_shapes",
                     b.getArrayAttr({b.getDenseI64ArrayAttr({2, 3}),
                                     b.getDenseI64ArrayAttr({-1}),
                                     b.getDenseI64ArrayAttr({2, 1})}));
        if (mode == 4)
          fields.set("input_shapes",
                     b.getArrayAttr({b.getDenseI64ArrayAttr({2, 3})}));
        if (mode == 5)
          fields.set("extra", b.getUnitAttr());
        if (mode == 6) {
          // Consistent counts and compatible shapes, but column IR now falsely
          // claims to implement row broadcasting. A product check cannot catch
          // it.
          fields.set("input_shapes",
                     b.getArrayAttr({b.getDenseI64ArrayAttr({2, 3}),
                                     b.getDenseI64ArrayAttr({2, 1}),
                                     b.getDenseI64ArrayAttr({2, 1})}));
          fields.set("input_counts", b.getDenseI64ArrayAttr({6, 2, 2}));
        }
        if (mode == 7)
          fields.set("output_count", b.getI32IntegerAttr(6));
        module->setAttr("sconeml.broadcast", fields.getDictionary(&context));
      });
    auto getLoad = [](mlir::ModuleOp module, unsigned number) {
      llvm::SmallVector<mlir::memref::LoadOp> loads;
      module.walk([&](mlir::memref::LoadOp load) { loads.push_back(load); });
      require(number < loads.size(), "missing broadcast load in fixture IR");
      return loads[number];
    };
    corrupt([&](auto module, auto &) {
      auto matrix = getLoad(module, 0), row = getLoad(module, 1);
      row->setOperand(1, matrix.getIndices().front());
    });
    corrupt([&](auto module, auto &) {
      getLoad(module, 1)->setOperand(0, getLoad(module, 2).getMemref());
    });
    for (int denominator : {0, 2, 4})
      corrupt([&](auto module, auto &b) {
        module.walk([&](mlir::arith::ConstantOp constant) {
          if (constant.getType().isIndex())
            constant.setValueAttr(b.getIndexAttr(denominator));
        });
      });
    corrupt([](auto module, auto &) {
      module.walk([](mlir::arith::DivUIOp divide) { divide.setIsExact(true); });
    });
    corrupt([](auto module, auto &b) {
      mlir::arith::DivUIOp divide;
      module.walk([&](mlir::arith::DivUIOp op) { divide = op; });
      b.setInsertionPoint(divide);
      auto wrong = mlir::arith::RemUIOp::create(
          b, divide.getLoc(), divide.getLhs(), divide.getRhs());
      divide.getResult().replaceAllUsesWith(wrong.getResult());
      divide.erase();
    });
    corrupt([](auto module, auto &) {
      mlir::arith::RemUIOp remainder;
      module.walk([&](mlir::arith::RemUIOp op) { remainder = op; });
      module.walk([&](mlir::arith::DivUIOp divide) {
        divide->setOperand(0, remainder.getResult());
      });
    });
    corrupt([](auto module, auto &) {
      module.walk([](mlir::arith::CmpIOp cmp) {
        cmp.setPredicate(mlir::arith::CmpIPredicate::slt);
      });
    });
    corrupt([&](auto module, auto &) {
      module.walk([&](mlir::memref::StoreOp store) {
        store->setOperand(2, getLoad(module, 2).getIndices().front());
      });
    });
    corrupt([&](auto module, auto &b) {
      auto row = getLoad(module, 1);
      b.setInsertionPoint(row);
      auto one = mlir::arith::ConstantIndexOp::create(b, row.getLoc(), 1);
      auto offset = mlir::arith::AddIOp::create(b, row.getLoc(),
                                                row.getIndices().front(), one);
      row->setOperand(1, offset);
    });
    corrupt([](auto module, auto &) {
      mlir::scf::IfOp guard;
      module.walk([&](mlir::scf::IfOp op) { guard = op; });
      auto constant = *guard.getThenRegion()
                           .front()
                           .template getOps<mlir::arith::ConstantOp>()
                           .begin();
      constant->moveBefore(guard);
    });
    corrupt([&](auto module, auto &b) {
      mlir::memref::StoreOp store;
      module.walk([&](mlir::memref::StoreOp op) { store = op; });
      b.setInsertionPoint(store);
      // Div/rem is legal only on the checked index path, never scalar i32 data.
      auto bad = mlir::arith::DivUIOp::create(
          b, store.getLoc(), getLoad(module, 0), getLoad(module, 1));
      store->setOperand(0, bad);
    });
    // Zero-output device IR must also contain no division/remainder, although
    // the public API skips extraction altogether for such calls.
    mlir::OwningOpRef<mlir::ModuleOp> empty =
        mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));
    std::array<ShapedTensorArgumentSpec, 2> emptySpecs{
        {{"a", 2, {2, 1}}, {"b", 0, {1, 0}}}};
    translateShapedTensorExpression(*empty, "a + b", emptySpecs);
    auto noWork = sconeml::lowerShapedTensorToGPU(*empty);
    require(mlir::succeeded(noWork), "empty broadcast extraction failed");
    bool division = false;
    (**noWork).walk([&](mlir::Operation *op) {
      division |= mlir::isa<mlir::arith::DivUIOp, mlir::arith::RemUIOp>(op);
    });
    auto noWorkExport = sconeml::exportMetalKernel(**noWork);
    require(!division && mlir::succeeded(noWorkExport) &&
                noWorkExport->shapedLayout->outputElementCount == 0 &&
                noWorkExport->shapedLayout->inputElementCounts ==
                    std::vector<size_t>({2, 0}),
            "empty broadcast generated unsafe indexing");
    const int64_t huge =
        static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) + 1;
    mlir::OwningOpRef<mlir::ModuleOp> large =
        mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));
    std::array<ShapedTensorArgumentSpec, 2> largeSpecs{
        {{"a", static_cast<size_t>(huge), {1, huge}}, {"b", 0, {0, 1}}}};
    translateShapedTensorExpression(*large, "7", largeSpecs);
    rejects("exceeds u32", [&] { sconeml::validateShapedTensorGPU(*large); });
    large = mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));
    const int64_t edge = std::numeric_limits<uint32_t>::max();
    largeSpecs = {
        {{"a", static_cast<size_t>(edge), {edge, 1}}, {"b", 2, {1, 2}}}};
    translateShapedTensorExpression(*large, "7", largeSpecs);
    rejects("exceeds u32", [&] { sconeml::validateShapedTensorGPU(*large); });
  }
  std::array<int32_t, 2> data{1, 2};
  std::array<ShapedTensorArgument, 2> emptyArgs{
      {{"a", data, {2, 1}}, {"b", {}, {1, 0}}}};
  std::string dump = "stale";
  const auto zero = runShapedTensorGPU("a + b", emptyArgs, &dump);
  require(zero.values.empty() && zero.shape == std::vector<int64_t>({2, 0}) &&
              dump.empty(),
          "empty API must skip shader/device");
#ifdef SCONEML_EXPECT_METAL_STUB
  rejects("only available on macOS",
          [&] { runShapedTensorGPU(test.program, args); });
#else
  if (gpuIsAvailable()) {
    const auto first = runShapedTensorGPU(test.program, args);
    require(first.values == expected(test) &&
                first.shape == std::vector<int64_t>({2, 3}),
            "direct broadcast value/shape mismatch");
    auto changed = test;
    changed.runs.front().arguments->at(1).values[0] = 99;
    const auto later = runShapedTensorGPU(changed.program, arguments(changed));
    require(later.values.front() == 200 && first.values == expected(test),
            "broadcast input values embedded or output invalidated");
    ShapedTensorResult escaped;
    {
      auto local = test;
      escaped = runShapedTensorGPU(local.program, arguments(local));
    }
    require(escaped.values == expected(test),
            "broadcast output did not outlive borrowed inputs");
    escaped.values.front() ^= 1;
    escaped.shape.front() = 7;
    require(args[0].values.front() == 1 && args[0].shape.front() == 2,
            "broadcast output aliases input");
    const auto overlap = sconeml::test::readCase(path, "overlap");
    auto overlapArgs = arguments(overlap);
    std::vector<int32_t> shared{1, 2, 3, 4, 5, 6, 999};
    const auto before = shared;
    overlapArgs[0].values = std::span<const int32_t>(shared).first(6);
    overlapArgs[1].values = std::span<const int32_t>(shared).subspan(1, 3);
    overlapArgs[2].values = std::span<const int32_t>(shared).first(2);
    require(runShapedTensorGPU(overlap.program, overlapArgs).values ==
                    expected(overlap) &&
                shared == before,
            "broadcast overlap/input canary failed");
    std::vector<MetalInput> buffers;
    for (const auto &arg : args)
      buffers.push_back(arg.values);
    std::vector<int32_t> output(8, 999);
    runShapedMetalKernel(owned.source, owned.entrypoint, *owned.shapedLayout,
                         buffers, output, 6, {4});
    require(std::vector<int32_t>(output.begin(), output.begin() + 6) ==
                    expected(test) &&
                output[6] == 999 && output[7] == 999,
            "unequal-buffer runtime readback or output canary failed");
    buffers[1] = buffers[1].first(2);
    rejects("input length", [&] {
      runShapedMetalKernel(owned.source, owned.entrypoint, *owned.shapedLayout,
                           buffers, output, 6);
    });
    require(output[6] == 999 && output[7] == 999,
            "rejected runtime launch touched output canary");
    llvm::outs() << "PASS broadcast GPU ownership/unequal buffers on "
                 << gpuName() << '\n';
  } else
    rejects("no Metal GPU", [&] { runShapedTensorGPU(test.program, args); });
#endif
  llvm::outs() << "PASS broadcast GPU metadata, bounds, empty and "
                  "malformed-index contracts\n";
}
