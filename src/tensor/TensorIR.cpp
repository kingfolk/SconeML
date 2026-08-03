#include "src/tensor/TensorIR.h"

#include "src/dialect/LetAlgDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"

#include "llvm/Support/raw_ostream.h"

#include <cctype>
#include <cstring>
#include <cstdlib>
#include <regex>
#include <stdexcept>
#include <utility>
#include <vector>

namespace sconeml::tensor {
namespace {

class ScalarExpressionParser {
public:
  ScalarExpressionParser(mlir::OpBuilder &builder, mlir::Location loc,
                         std::string parameter, mlir::Value parameterValue,
                         const std::string &source)
      : builder(builder), loc(loc), parameter(std::move(parameter)),
        parameterValue(parameterValue), tokens(tokenize(source)) {}

  mlir::Value parse() {
    mlir::Value value = parseAddSub();
    if (position != tokens.size())
      throw std::runtime_error("unexpected token in tensor expression: " +
                               tokens[position]);
    return value;
  }

private:
  static std::vector<std::string> tokenize(const std::string &source) {
    std::vector<std::string> result;
    for (size_t i = 0; i < source.size();) {
      if (std::isspace(static_cast<unsigned char>(source[i]))) {
        ++i;
      } else if (std::strchr("+-*()", source[i])) {
        result.emplace_back(1, source[i++]);
      } else if (std::isalpha(static_cast<unsigned char>(source[i])) ||
                 source[i] == '_') {
        size_t begin = i++;
        while (i < source.size() &&
               (std::isalnum(static_cast<unsigned char>(source[i])) ||
                source[i] == '_'))
          ++i;
        result.push_back(source.substr(begin, i - begin));
      } else if (std::isdigit(static_cast<unsigned char>(source[i])) ||
                 source[i] == '.') {
        size_t begin = i++;
        while (i < source.size() &&
               (std::isdigit(static_cast<unsigned char>(source[i])) ||
                source[i] == '.'))
          ++i;
        result.push_back(source.substr(begin, i - begin));
      } else {
        throw std::runtime_error("unsupported character in tensor expression");
      }
    }
    return result;
  }

  mlir::Value parseAddSub() {
    mlir::Value value = parseMultiply();
    while (position < tokens.size() &&
           (tokens[position] == "+" || tokens[position] == "-")) {
      const std::string op = tokens[position++];
      mlir::Value rhs = parseMultiply();
      if (op == "+")
        value = mlir::arith::AddFOp::create(builder, loc, value, rhs);
      else
        value = mlir::arith::SubFOp::create(builder, loc, value, rhs);
    }
    return value;
  }

  mlir::Value parseMultiply() {
    mlir::Value value = parsePrimary();
    while (position < tokens.size() && tokens[position] == "*") {
      ++position;
      value = mlir::arith::MulFOp::create(builder, loc, value, parsePrimary());
    }
    return value;
  }

  mlir::Value parsePrimary() {
    if (position == tokens.size())
      throw std::runtime_error("unexpected end of tensor expression");
    const std::string token = tokens[position++];
    if (token == "(") {
      mlir::Value value = parseAddSub();
      if (position == tokens.size() || tokens[position++] != ")")
        throw std::runtime_error("missing ')' in tensor expression");
      return value;
    }
    if (token == parameter)
      return parameterValue;
    char *end = nullptr;
    const float number = std::strtof(token.c_str(), &end);
    if (end && *end == '\0')
      return mlir::arith::ConstantOp::create(
          builder, loc, builder.getF32FloatAttr(number));
    throw std::runtime_error("unknown tensor expression variable: " + token);
  }

  mlir::OpBuilder &builder;
  mlir::Location loc;
  std::string parameter;
  mlir::Value parameterValue;
  std::vector<std::string> tokens;
  size_t position = 0;
};

struct TensorFunctionSource {
  std::string name;
  std::string parameter;
  std::string expression;
};

TensorFunctionSource parseTensorFunction(const std::string &source) {
  static const std::regex declaration(
      R"(^\s*let\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*:\s*tensor\s*<\s*f32\s*>\s*\)\s*=\s*([\s\S]*?)\s*$)");
  std::smatch match;
  if (!std::regex_match(source, match, declaration))
    throw std::runtime_error(
        "tensor programs must use: let name (x : tensor<f32>) = expression");
  return {match[1].str(), match[2].str(), match[3].str()};
}

std::unique_ptr<mlir::MLIRContext> createTensorContext() {
  mlir::DialectRegistry registry;
  mlir::registerAllToLLVMIRTranslations(registry);
  auto context = std::make_unique<mlir::MLIRContext>(registry);
  context->getOrLoadDialect<sconeml::letalg::LetAlgDialect>();
  context->getOrLoadDialect<mlir::arith::ArithDialect>();
  context->getOrLoadDialect<mlir::cf::ControlFlowDialect>();
  context->getOrLoadDialect<mlir::func::FuncDialect>();
  context->getOrLoadDialect<mlir::LLVM::LLVMDialect>();
  context->getOrLoadDialect<mlir::memref::MemRefDialect>();
  context->getOrLoadDialect<mlir::scf::SCFDialect>();
  return context;
}

} // namespace

TensorModule translateTensorProgram(const std::string &source) {
  const TensorFunctionSource program = parseTensorFunction(source);
  auto context = createTensorContext();

  mlir::OpBuilder builder(context.get());
  mlir::Location loc = builder.getUnknownLoc();
  auto module = mlir::ModuleOp::create(builder, loc);
  builder.setInsertionPointToStart(module.getBody());

  auto bufferType = mlir::MemRefType::get(
      {mlir::ShapedType::kDynamic}, builder.getF32Type());
  auto functionType = builder.getFunctionType({bufferType, bufferType}, {});
  auto function = mlir::func::FuncOp::create(
      builder, loc, program.name, functionType);
  function->setAttr("llvm.emit_c_interface", builder.getUnitAttr());
  mlir::Block *entry = function.addEntryBlock();
  builder.setInsertionPointToStart(entry);

  auto map = sconeml::letalg::TensorMapOp::create(
      builder, loc, entry->getArgument(0), entry->getArgument(1));
  mlir::Block *body = builder.createBlock(
      &map.getBody(), {}, {builder.getF32Type()}, {loc});
  builder.setInsertionPointToStart(body);
  mlir::Value result = ScalarExpressionParser(
      builder, loc, program.parameter, body->getArgument(0), program.expression)
                           .parse();
  sconeml::letalg::YieldOp::create(builder, loc, result);

  builder.setInsertionPointToEnd(entry);
  mlir::func::ReturnOp::create(builder, loc);

  if (mlir::failed(mlir::verify(module)))
    throw std::runtime_error("translated tensor LetAlg module failed verification");
  return {std::move(context), std::move(module)};
}

std::string printModule(mlir::Operation *operation) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  operation->print(stream);
  return text;
}

std::string translateTensorIR(const std::string &source) {
  TensorModule tensorModule = translateTensorProgram(source);
  return printModule(tensorModule.module.get());
}

} // namespace sconeml::tensor
