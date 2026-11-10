#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "src/tensor/CPURunner.h"
#include "src/tensor/TensorFrontend.h"
#include "test/runner/TestSpec.h"
#include <array>
#include <bit>
#include <cfenv>
#include <iostream>

using namespace sconeml::tensor;
using namespace sconeml::test;
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
Number number(const std::vector<float> &values) {
  Number n{true, {}, true, {}};
  for (float f : values)
    n.f32Bits.push_back(std::bit_cast<uint32_t>(f));
  return n;
}
} // namespace
int main(int argc, char **argv) {
  try {
    require(argc == 2, "f32 fixture required");
    const auto fixture = readCase(argv[1], "row-broadcast");
    std::vector<std::vector<float>> storage;
    for (const auto &a : *fixture.runs.front().arguments)
      storage.push_back(f32Values(a));
    const auto original = storage;
    std::vector<ShapedTensorF32Argument> args;
    for (size_t i = 0; i < storage.size(); ++i) {
      const auto &a = fixture.runs.front().arguments->at(i);
      args.push_back({a.name, storage[i], *a.shape});
    }
    Number expected;
    for (const auto &a : fixture.assertions.at("cpu"))
      if (a.kind == "result")
        expected = parseNumber(a.value, true);
    std::string ir, changedIR;
    auto result = runShapedTensorF32CPU(fixture.program, args, &ir);
    require(number(result.values) == expected &&
                result.shape == std::vector<int64_t>({2, 3}) &&
                storage == original,
            "direct f32 ABI/input immutability");
    storage.front().front() = 8.0f;
    const auto changed =
        runShapedTensorF32CPU(fixture.program, args, &changedIR);
    require(changed.values.front() == 4.25f && ir == changedIR &&
                number(result.values) == expected,
            "values entered IR or output did not own storage");
    storage.clear();
    args.clear();
    require(number(result.values) == expected,
            "output did not survive inputs/JIT");

    std::vector<float> data{1, 2, 3, 4, 5, 6};
    const auto saved = data;
    std::array<ShapedTensorF32Argument, 2> overlap{
        {{"a", data, {2, 3}},
         {"b", std::span<const float>(data).subspan(1, 3), {3}}}};
    auto shared = runShapedTensorF32CPU(fixture.program, overlap);
    // Compare actual aliasing to the same fixture program with disjoint copies.
    std::vector<float> row{2, 3, 4};
    auto disjoint = overlap;
    disjoint[1].values = row;
    require(shared.values ==
                    runShapedTensorF32CPU(fixture.program, disjoint).values &&
                data == saved,
            "overlapping broadcast spans");
    auto alias = overlap;
    alias[1] = {"b", data, {2, 3}};
    auto copies = alias;
    std::vector<float> second = data;
    copies[1].values = second;
    require(runShapedTensorF32CPU(fixture.program, alias).values ==
                runShapedTensorF32CPU(fixture.program, copies).values,
            "identical input alias");
    result.values.front() = 99;
    result.shape.front() = 99;
    require(data == saved, "result aliases input");

    std::array<ShapedTensorF32Argument, 1> empty{{{"a", {}, {0, 3}}}};
    ir = "stale";
    auto zero = runShapedTensorF32CPU("a * 0.5", empty, &ir);
    require(zero.values.empty() && zero.shape == empty[0].shape && ir.empty(),
            "empty f32 must skip JIT");
    rejects("missing an operand", [&] { runShapedTensorF32CPU("a +", empty); });
    auto bad = empty;
    bad[0].shape = {1};
    rejects("storage count", [&] { runShapedTensorF32CPU("a", bad); });
    const auto rounding = std::fegetround();
    std::fesetround(FE_DOWNWARD);
    rejects("round-to-nearest", [&] { runShapedTensorF32CPU("a", empty); });
    std::fesetround(rounding);

    mlir::MLIRContext context;
    context.getOrLoadDialect<mlir::arith::ArithDialect>();
    context.getOrLoadDialect<mlir::func::FuncDialect>();
    context.getOrLoadDialect<mlir::memref::MemRefDialect>();
    context.getOrLoadDialect<mlir::linalg::LinalgDialect>();
    auto build = [&] {
      auto m = mlir::OwningOpRef<mlir::ModuleOp>(
          mlir::ModuleOp::create(mlir::UnknownLoc::get(&context)));
      std::array<ShapedTensorArgumentSpec, 1> specs{{{"a", 1, {1}}}};
      translateShapedTensorF32Expression(*m, "a * 0.5", specs);
      return m;
    };
    auto corrupt = [&](auto action) {
      auto m = build();
      action(*m);
      rejects("invalid shaped tensor IR",
              [&] { validateShapedTensorExpression(*m); });
    };
    corrupt([&](auto m) {
      auto f = *m.template getOps<mlir::func::FuncOp>().begin();
      f->removeAttr("sconeml.float_profile");
    });
    corrupt([&](auto m) {
      m.walk([&](mlir::arith::MulFOp op) {
        op.setFastmath(mlir::arith::FastMathFlags::contract);
      });
    });
    corrupt([&](auto m) {
      m.walk([&](mlir::linalg::GenericOp op) {
        auto &b = op.getRegion().front();
        b.getTerminator()->setOperand(0, b.getArgument(0));
      });
    });
    corrupt([&](auto m) {
      m.walk([&](mlir::arith::ConstantOp op) {
        if (auto a = mlir::dyn_cast<mlir::IntegerAttr>(op.getValue());
            a && a.getInt() == 0x7fffffff)
          op.setValueAttr(mlir::IntegerAttr::get(a.getType(), 0));
      });
    });

    auto spec = std::string(
        "(* @test\nversion: 1\ninput: source\nruns:\n  - name: cpu\n    "
        "pipeline: shaped-f32-cpu\n    profile: f32-ftz\n    arguments:\n      "
        "- name: a\n        dtype: f32\n        values: [0.5]\n        shape: "
        "[1]\n*)\n(* @assert:cpu:result\n[0.5]\n*)\n(* "
        "@assert:cpu:shape\n[1]\n*)\na");
    auto mutated = [&](const std::string &from, const std::string &to) {
      auto s = spec;
      s.replace(s.find(from), from.size(), to);
      return s;
    };
    rejects("profile",
            [&] { parseCase("config.ml", mutated("f32-ftz", "ieee")); });
    rejects("profile", [&] {
      parseCase("config.ml", mutated("    profile: f32-ftz\n", ""));
    });
    rejects("dtype", [&] {
      parseCase("config.ml", mutated("dtype: f32", "dtype: f16"));
    });
    rejects("invalid f32", [&] {
      parseCase("config.ml", mutated("values: [0.5]", "values: [oops]"));
    });
    auto plusZero = parseNumber("[0x00000000]", true),
         minusZero = parseNumber("[0x80000000]", true);
    require(!(plusZero == minusZero), "float assertions lost zero sign");
    require(parseNumber("[0x7fc12345]", true) == parseNumber("[nan]", true),
            "NaN classification assertion");
    require(!(parseNumber("[inf]", true) == parseNumber("[-inf]", true)),
            "infinity sign assertion");
    std::cout << "f32 CPU ownership, ABI, FTZ IR and schema contracts passed\n";
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
