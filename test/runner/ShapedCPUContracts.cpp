#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "src/tensor/CPURunner.h"
#include "src/tensor/TensorFrontend.h"
#include "test/runner/TestSpec.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
#include <limits>
#include <stdexcept>

using namespace sconeml::tensor;
using sconeml::test::require;
namespace {
template <typename F> void rejects(const char *message, F action) {
  try {
    action();
  } catch (const std::exception &error) {
    require(std::string(error.what()).find(message) != std::string::npos,
            std::string("wrong rejection: ") + error.what());
    return;
  }
  throw std::runtime_error(std::string("accepted: ") + message);
}
std::vector<ShapedTensorArgument>
arguments(const sconeml::test::TestCase &test) {
  std::vector<ShapedTensorArgument> result;
  for (const auto &input : *test.runs.front().arguments)
    result.push_back({input.name, input.values, *input.shape});
  return result;
}
std::vector<int32_t> expected(const sconeml::test::TestCase &test) {
  for (const auto &assertion : test.assertions.at("cpu"))
    if (assertion.kind == "result")
      return sconeml::test::parseNumber(assertion.value).values;
  throw std::runtime_error("missing fixture expectation");
}
std::string print(mlir::ModuleOp module) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  module.print(stream);
  return text;
}
} // namespace

int main(int argc, char **argv) {
  try {
    require(argc == 2, "usage: shaped_cpu_contracts <same_shape.ml>");
    auto test = sconeml::test::readCase(argv[1], "rank2-three");
    auto args = arguments(test);
    std::string beforeIR, changedIR;
    const auto before = runShapedTensorCPU(test.program, args, &beforeIR);
    require(before.values == expected(test), "direct three-input ABI result");
    auto changed = test;
    changed.runs.front().arguments->front().values.front() = 8;
    auto changedArgs = arguments(changed);
    const auto later =
        runShapedTensorCPU(changed.program, changedArgs, &changedIR);
    require(later.values.front() == 172 && later.values != before.values &&
                beforeIR == changedIR && before.values == expected(test),
            "host values entered IR or later JIT invalidated earlier output");
    for (auto &arg : changedArgs)
      arg.shape = {3, 2};
    const auto reshaped =
        runShapedTensorCPU(changed.program, changedArgs, &changedIR);
    require(reshaped.values == later.values &&
                reshaped.shape == std::vector<int64_t>({3, 2}) &&
                beforeIR != changedIR,
            "shape did not specialize view/loop IR");

    ShapedTensorResult escaped;
    {
      auto local = test;
      auto localArgs = arguments(local);
      escaped = runShapedTensorCPU(local.program, localArgs);
      local.runs.front().arguments->front().values.front() = 999;
      localArgs.front().shape = {6};
    }
    require(escaped.values == expected(test) &&
                escaped.shape == std::vector<int64_t>({2, 3}),
            "result did not own data/shape after input and JIT destruction");
    escaped.values.front() ^= 1;
    escaped.shape.front() = 99;
    require(args.front().values.front() == 1 && args.front().shape.front() == 2,
            "output aliases caller data/shape");

    const auto alias = sconeml::test::readCase(argv[1], "alias");
    auto aliasArgs = arguments(alias);
    aliasArgs[1].values = aliasArgs[0].values;
    require(runShapedTensorCPU(alias.program, aliasArgs).values ==
                expected(alias),
            "read-only input alias failed");
    const auto overlap = sconeml::test::readCase(argv[1], "overlap");
    auto overlapArgs = arguments(overlap);
    std::vector<int32_t> storage{1, 2, 3, 4, 5, 6, 7, 8};
    const auto original = storage;
    overlapArgs[0].values = std::span<const int32_t>(storage).first(6);
    overlapArgs[1].values = std::span<const int32_t>(storage).subspan(2, 6);
    require(runShapedTensorCPU(overlap.program, overlapArgs).values ==
                    expected(overlap) &&
                storage == original,
            "overlap or input immutability failed");

    std::array<ShapedTensorArgument, 1> empty{{{"a", {}, {0, 3}}}};
    std::string dump = "stale";
    auto zero = runShapedTensorCPU("a + 1", empty, &dump);
    require(zero.values.empty() && zero.shape == empty.front().shape &&
                dump.empty(),
            "empty execution must preserve shape and skip JIT/dump");
    rejects("empty or missing an operand",
            [&] { runShapedTensorCPU("", empty); });
    // Old APIs must retain equal-length behavior (no implicit broadcast).
    std::array<TensorArgument, 2> flat{
        {{"a", args[0].values}, {"b", args[0].values.first(1)}}};
    rejects("matching lengths", [&] { runTensorCPU("a + b", flat); });

    mlir::MLIRContext context;
    context.getOrLoadDialect<mlir::arith::ArithDialect>();
    context.getOrLoadDialect<mlir::func::FuncDialect>();
    context.getOrLoadDialect<mlir::memref::MemRefDialect>();
    context.getOrLoadDialect<mlir::linalg::LinalgDialect>();
    auto build = [&] {
      mlir::OwningOpRef<mlir::ModuleOp> module =
          mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));
      std::array<ShapedTensorArgumentSpec, 2> specs{
          {{"a", 6, {2, 3}}, {"b", 6, {2, 3}}}};
      translateShapedTensorExpression(*module, "a + b", specs);
      return module;
    };
    auto baseline = build();
    const auto originalIR = print(*baseline);
    validateShapedTensorExpression(*baseline);
    require(print(*baseline) == originalIR, "validation mutated module");
    auto corrupt = [&](auto mutation) {
      auto module = build();
      auto function = *module->getOps<mlir::func::FuncOp>().begin();
      auto map = *function.getOps<mlir::linalg::GenericOp>().begin();
      mutation(function, map);
      rejects("invalid shaped tensor IR",
              [&] { validateShapedTensorExpression(*module); });
    };
    corrupt([&](auto, auto map) {
      auto &body = map.getRegion().front();
      body.front().setOperand(0, body.getArguments().back());
    });
    corrupt(
        [&](auto, auto map) { map->setOperand(0, map.getOutputs().front()); });
    corrupt([&](auto, auto map) {
      auto maps = map.getIndexingMapsArray();
      maps[0] = mlir::AffineMap::get(2, 0,
                                     {mlir::getAffineConstantExpr(0, &context),
                                      mlir::getAffineDimExpr(1, &context)},
                                     &context);
      mlir::OpBuilder builder(&context);
      map.setIndexingMapsAttr(builder.getAffineMapArrayAttr(maps));
    });
    corrupt([&](auto function, auto) {
      auto view =
          *function.template getOps<mlir::memref::ReinterpretCastOp>().begin();
      mlir::OpBuilder builder(&context);
      view.setStaticOffsetsAttr(builder.getDenseI64ArrayAttr({1}));
    });
    corrupt([&](auto function, auto map) {
      mlir::OpBuilder builder(function);
      builder.setInsertionPoint(map);
      auto capture =
          mlir::arith::ConstantIntOp::create(builder, map.getLoc(), 7, 32);
      map.getRegion().front().front().setOperand(0, capture);
    });
    corrupt([&](auto, auto map) {
      auto add =
          mlir::cast<mlir::arith::AddIOp>(&map.getRegion().front().front());
      add.setOverflowFlags(mlir::arith::IntegerOverflowFlags::nsw);
    });
    corrupt([&](auto, auto map) {
      auto &body = map.getRegion().front();
      mlir::OpBuilder builder(&body.front());
      auto div = mlir::arith::DivSIOp::create(
          builder, map.getLoc(), body.getArgument(0), body.getArgument(1));
      body.front().getNextNode()->setOperand(0, div);
    });
    // Exercise capacity/overflow through metadata, without forged
    // spans/allocations.
    std::array<ShapedTensorArgumentSpec, 1> huge{
        {{"a", std::numeric_limits<size_t>::max(), {1}}}};
    rejects("storage count",
            [&] { translateShapedTensorExpression(*baseline, "a", huge); });
    huge[0] = {"a", 0, {0, std::numeric_limits<int64_t>::max()}};
    rejects("host index/byte/capacity",
            [&] { translateShapedTensorExpression(*baseline, "a", huge); });
    llvm::outs() << "PASS shaped CPU ABI, specialization, ownership, empty, "
                    "bounds and IR contracts\n";
    return 0;
  } catch (const std::exception &error) {
    llvm::errs() << "FAIL: " << error.what() << '\n';
    return 1;
  }
}
