#include "test/runner/SumExecution.h"
#include "src/tensor/GPURunner.h"
#include "src/tensor/Sum.h"
#include "src/tensor/SumGPU.h"
#include <bit>
#include <cstring>
namespace sconeml::test {
Output executeSum(const TestCase &test, const Run &run, Output &out,
                  bool disableMetal) {
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
    const bool gpu =
        run.pipeline.ends_with("-metal") || run.pipeline.ends_with("-msl");
    if (gpu) {
      out.stage = "gpu-lowering";
      auto program = tensor::exportSumMetal(
          plan, f32, {run.threadsPerThreadgroup.value_or(0)});
      out.text = program.source;
      out.entrypoint = "sum_tree";
      if (run.pipeline.ends_with("-msl")) {
        out.success = true;
        return out;
      }
      if (!program.dispatches.empty() &&
          (disableMetal || !tensor::gpuIsAvailable())) {
        out.stage = "metal-execute";
        out.skipped = true;
        out.diagnostic = "Metal device unavailable or disabled";
        return out;
      }
    }
    out.stage = gpu ? "metal-execute" : "cpu-execute";
    if (f32) {
      auto result =
          gpu ? tensor::runSumF32GPU(test.program, args, &out.text,
                                     {run.threadsPerThreadgroup.value_or(0)})
              : tensor::runSumF32CPU(test.program, args, &out.ir);
      Number number{true, {}, true, {}};
      for (auto v : result.values)
        number.f32Bits.push_back(std::bit_cast<uint32_t>(v));
      out.number = std::move(number);
      out.shape = std::move(result.shape);
    } else {
      auto result =
          gpu ? tensor::runSumI32GPU(test.program, ints, &out.text,
                                     {run.threadsPerThreadgroup.value_or(0)})
              : tensor::runSumI32CPU(test.program, ints, &out.ir);
      out.number = Number{true, std::move(result.values)};
      out.shape = std::move(result.shape);
    }
  }
  out.success = true;
  return out;
}
} // namespace sconeml::test
