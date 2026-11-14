#include "test/runner/SumExecution.h"
#include "src/tensor/Sum.h"
#include <bit>
#include <cstring>
namespace sconeml::test {
Output executeSum(const TestCase &test, const Run &run, Output &out, bool) {
  out.stage = "frontend";
  const bool f32 = run.pipeline.starts_with("sum-f32-");
  std::vector<std::vector<float>> floats;
  std::vector<tensor::ShapedTensorArgumentSpec> specs;
  std::vector<tensor::ShapedTensorArgument> ints;
  std::vector<tensor::ShapedTensorF32Argument> args;
  for (const auto &a : *run.arguments)
    if (f32)
      floats.push_back(f32Values(a));
  for (size_t i = 0; i < run.arguments->size(); ++i) {
    const auto &a = run.arguments->at(i);
    specs.push_back(
        {a.name, f32 ? floats[i].size() : a.values.size(), *a.shape});
    if (f32)
      args.push_back({a.name, floats[i], *a.shape});
    else
      ints.push_back({a.name, a.values, *a.shape});
  }
  auto plan = tensor::prepareSum(test.program, specs, f32);
  if (run.pipeline.ends_with("-ir")) {
    auto ctx = makeContext();
    auto m = mlir::OwningOpRef<mlir::ModuleOp>(
        mlir::ModuleOp::create(mlir::UnknownLoc::get(ctx.get())));
    tensor::buildSumCPU(*m, plan, f32);
    llvm::raw_string_ostream stream(out.ir);
    m->print(stream);
  } else {
    out.stage = "cpu-execute";
    if (f32) {
      auto result = tensor::runSumF32CPU(test.program, args, &out.ir);
      Number number{true, {}, true, {}};
      for (auto v : result.values)
        number.f32Bits.push_back(std::bit_cast<uint32_t>(v));
      out.number = std::move(number);
      out.shape = std::move(result.shape);
    } else {
      auto result = tensor::runSumI32CPU(test.program, ints, &out.ir);
      out.number = Number{true, std::move(result.values)};
      out.shape = std::move(result.shape);
    }
  }
  out.success = true;
  return out;
}
} // namespace sconeml::test
