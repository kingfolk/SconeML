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
#include <sstream>
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
        value = mlir::arith::AddIOp::create(builder, loc, value, rhs);
      else
        value = mlir::arith::SubIOp::create(builder, loc, value, rhs);
    }
    return value;
  }

  mlir::Value parseMultiply() {
    mlir::Value value = parsePrimary();
    while (position < tokens.size() && tokens[position] == "*") {
      ++position;
      mlir::Value rhs = parsePrimary();
      value = mlir::arith::MulIOp::create(builder, loc, value, rhs);
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
    const long number = std::strtol(token.c_str(), &end, 10);
    if (end && *end == '\0')
      return mlir::arith::ConstantOp::create(
          builder, loc, builder.getI32IntegerAttr(number));
    throw std::runtime_error("unknown tensor expression variable: " + token);
  }

  mlir::OpBuilder &builder;
  mlir::Location loc;
  std::string parameter;
  mlir::Value parameterValue;
  std::vector<std::string> tokens;
  size_t position = 0;
};

struct TensorLiteralSource {
  std::string name;
  std::string variable;
  std::vector<int32_t> elements;
  std::string expression;
};

TensorLiteralSource parseTensorLiteral(const std::string &source) {
  static const std::regex declaration(
      R"(^\s*let\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*\[([^\]]*)\]\s*in\s*([\s\S]*?)\s*$)");
  std::smatch match;
  if (!std::regex_match(source, match, declaration))
    throw std::runtime_error(
        "tensor literals must use: let name = [1, 2, ...] in expression");

  std::vector<int32_t> elements;
  std::istringstream values(match[2].str());
  std::string token;
  while (std::getline(values, token, ',')) {
    token.erase(0, token.find_first_not_of(" \t\n\r"));
    token.erase(token.find_last_not_of(" \t\n\r") + 1);
    char *end = nullptr;
    const long value = std::strtol(token.c_str(), &end, 10);
    if (token.empty() || !end || *end != '\0')
      throw std::runtime_error("tensor literal elements must be i32 integers");
    elements.push_back(static_cast<int32_t>(value));
  }
  if (elements.empty())
    throw std::runtime_error("tensor literals must contain at least one element");
  return {"tensor_literal", match[1].str(), std::move(elements), match[3].str()};
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
  const TensorLiteralSource program = parseTensorLiteral(source);
  auto context = createTensorContext();
  mlir::OpBuilder builder(context.get());
  mlir::Location loc = builder.getUnknownLoc();
  auto module = mlir::ModuleOp::create(builder, loc);
  builder.setInsertionPointToStart(module.getBody());

  auto bufferType = mlir::MemRefType::get(
      {static_cast<int64_t>(program.elements.size())}, builder.getI32Type());
  auto functionType = builder.getFunctionType({}, {bufferType});
  auto function = mlir::func::FuncOp::create(builder, loc, program.name, functionType);
  mlir::Block *entry = function.addEntryBlock();
  builder.setInsertionPointToStart(entry);
  auto input = mlir::memref::AllocOp::create(builder, loc, bufferType);
  auto output = mlir::memref::AllocOp::create(builder, loc, bufferType);
  for (size_t i = 0; i < program.elements.size(); ++i) {
    auto index = mlir::arith::ConstantIndexOp::create(builder, loc, i);
    auto value = mlir::arith::ConstantOp::create(
        builder, loc, builder.getI32IntegerAttr(program.elements[i]));
    mlir::memref::StoreOp::create(builder, loc, value, input.getResult(),
                                  mlir::ValueRange{index.getResult()});
  }

  auto map = sconeml::letalg::TensorMapOp::create(
      builder, loc, input.getResult(), output.getResult());
  mlir::Block *body = builder.createBlock(
      &map.getBody(), {}, {builder.getI32Type()}, {loc});
  builder.setInsertionPointToStart(body);
  mlir::Value result = ScalarExpressionParser(
      builder, loc, program.variable, body->getArgument(0), program.expression)
                           .parse();
  sconeml::letalg::YieldOp::create(builder, loc, result);

  builder.setInsertionPointToEnd(entry);
  mlir::func::ReturnOp::create(builder, loc, mlir::ValueRange{output.getResult()});

  if (mlir::failed(mlir::verify(module)))
    throw std::runtime_error("translated tensor literal module failed verification");
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
