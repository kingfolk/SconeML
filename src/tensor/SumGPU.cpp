#include "src/tensor/SumGPU.h"
#include <stdexcept>
namespace sconeml::tensor {
SumMetalProgram exportSumMetal(const SumPlan &plan, bool f32,
                               MetalLaunchOptions options) {
  validateSumPlan(plan);
  if (options.threadsPerThreadgroup && options.threadsPerThreadgroup != 256)
    throw std::invalid_argument(
        "sum numerical contract requires threadgroup width 256");
  SumMetalProgram p;
  p.plan = plan;
  p.dtype = f32 ? MetalElementType::F32 : MetalElementType::I32;
  if (f32)
    p.profile = "f32-sum-tree256-ftz";
  if (!plan.outputCount || !plan.length)
    return p;
  size_t length = plan.length, base = plan.baseStride,
         stride = plan.inputStride;
  do {
    auto parts = length / 256 + (length % 256 != 0);
    p.dispatches.push_back(
        {length, parts, plan.outputCount * parts, base, stride});
    length = parts;
    base = parts;
    stride = 1;
  } while (length > 1);
  p.source = "#include <metal_stdlib>\nusing namespace metal;\n#pragma STDC "
             "FP_CONTRACT OFF\n";
  if (f32)
    p.source += R"(using Element = float;
inline float sum_normalize(float v) {
  uint bits = as_type<uint>(v);
  return as_type<float>((bits & 0x7fffffffu) < 0x00800000u ? 0u : bits);
}
inline float sum_add(float a, float b) { return sum_normalize(a + b); }
)";
  else
    p.source +=
        "using Element = uint;\ninline uint sum_normalize(uint v) { return v; "
        "}\ninline uint sum_add(uint a, uint b) { return a + b; }\n";
  p.source += R"(kernel void sum_tree(device const Element *input [[buffer(0)]],
                     device Element *output [[buffer(1)]],
                     constant uint &length [[buffer(2)]],
                     constant uint &parts [[buffer(3)]],
                     constant uint &base_stride [[buffer(4)]],
                     constant uint &input_stride [[buffer(5)]],
                     uint group [[threadgroup_position_in_grid]],
                     uint lane [[thread_index_in_threadgroup]]) {
  threadgroup Element scratch[256];
  const uint row = group / parts;
  const uint chunk = group % parts;
  const uint k = chunk * 256u + lane;
  scratch[lane] = k < length ? sum_normalize(input[row * base_stride + k * input_stride]) : Element(0);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint stride = 1u; stride < 256u; stride *= 2u) {
    if (lane % (2u * stride) == 0u)
      scratch[lane] = sum_add(scratch[lane], scratch[lane + stride]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (lane == 0u) output[group] = scratch[0];
}
)";
  return p;
}
void validateSumMetal(const SumMetalProgram &p, MetalElementType actual,
                      size_t in, size_t out, size_t maxBytes) {
  validateSumPlan(p.plan);
  if (p.dtype != actual ||
      (p.dtype != MetalElementType::I32 && p.dtype != MetalElementType::F32))
    throw std::invalid_argument("sum buffer dtype does not match metadata");
  auto canonical = exportSumMetal(p.plan, p.dtype == MetalElementType::F32);
  if (p.width != canonical.width || p.elementBytes != canonical.elementBytes ||
      p.profile != canonical.profile || p.source != canonical.source ||
      p.dispatches != canonical.dispatches)
    throw std::invalid_argument(
        "sum Metal metadata/source/dispatch dependencies are not canonical");
  if (in != p.plan.inputCount || out != p.plan.outputCount)
    throw std::invalid_argument("sum buffer lengths do not match metadata");
  auto bytes = [&](size_t n) {
    if (n > std::numeric_limits<size_t>::max() / 4 || n * 4 > maxBytes)
      throw std::invalid_argument(
          "sum buffer byte size exceeds supported limit");
  };
  bytes(in);
  bytes(out);
  for (const auto &d : p.dispatches)
    bytes(d.groupCount);
}
namespace {
template <typename T, typename A>
SumResult<T> run(const std::string &source, std::span<const A> args,
                 std::string *dump, MetalLaunchOptions options) {
  if (dump)
    dump->clear();
  constexpr bool f32 = std::is_same_v<T, float>;
  std::vector<ShapedTensorArgumentSpec> specs;
  for (const auto &a : args) {
    if (!a.values.empty() && !a.values.data())
      throw std::invalid_argument(
          "sum nonempty input requires nonnull storage");
    specs.push_back({a.name, a.values.size(), a.shape});
  }
  auto plan = prepareSum(source, specs, f32);
  auto program = exportSumMetal(plan, f32, options);
  auto type = f32 ? MetalElementType::F32 : MetalElementType::I32;
  validateSumMetal(program, type, args.front().values.size(), plan.outputCount);
  SumResult<T> result{std::vector<T>(plan.outputCount), plan.outputShape};
  if (dump)
    *dump = program.source;
  detail::runSumMetalBytes(program, type, std::as_bytes(args.front().values),
                           std::as_writable_bytes(std::span<T>(result.values)));
  return result;
}
} // namespace
SumI32Result runSumI32GPU(const std::string &s,
                          std::span<const ShapedTensorArgument> a,
                          std::string *msl, MetalLaunchOptions o) {
  return run<int32_t>(s, a, msl, o);
}
SumF32Result runSumF32GPU(const std::string &s,
                          std::span<const ShapedTensorF32Argument> a,
                          std::string *msl, MetalLaunchOptions o) {
  return run<float>(s, a, msl, o);
}
} // namespace sconeml::tensor
