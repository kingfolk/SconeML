#ifndef SCONEML_PARSER_H
#define SCONEML_PARSER_H

#include <memory>
#include <string>
#include <unordered_set>
#include <functional>
#include <charconv>
#include <cstdint>
#include <cctype>
#include <string_view>
#include <memory>
#include "Ast.h"

namespace sconeml {

void tokenize(std::string& input, std::vector<std::string>& tokens) {
  std::string tok;
  for (auto c : input) {
    if (c == ' ' || c == '\n' || c == ';') {
      if (tok.size() > 0) tokens.push_back(tok);
      tok.resize(0);
    } else if (c == '(' || c == ')' || c == '{' || c == '}' || c == '[' ||
               c == ']' || c == ',' || c == '+' || c == '-' || c == '*' ||
               c == '/' || c == '^' || c == '!' || c == '~' || c == '>' ||
               c == '<') {
      if (tok.size() > 0) tokens.push_back(tok);
      tokens.push_back(std::string{c});
      tok.resize(0);
    } else {
      tok.push_back(c);
    }
  }
  if (tok.size() > 0) {
    tokens.push_back(tok);
  }
}

// This is a roughly crafted LL style parser. It's not very carefully designed but just used for proof 
// of concept for the later MLIR logics. It's not very important to generate all correct AST node for
// every raw input, but generate some correct AST node for later MLIR codegen part.
std::unique_ptr<ExprNode> parse(std::string& input) {
  std::vector<std::string> tokens;
  tokenize(input, tokens);
  printf("===== tokens =====\n");
  for (auto& t : tokens) {
    printf("%s, ", t.c_str());
  }
  printf("\n");

  std::unordered_set<std::string> fns;
  auto reduceOpt = [&](std::vector<std::unique_ptr<ExprNode>>& stack, std::vector<char>& operators) {
    // printf("[reduceOpt] %lu %lu\n", operators.size(), stack.size());
    int start = 0;
    if (stack[start]->getKind() == ExprNode::Kind_Var && fns.contains(stack[start]->dump())) {
      auto fn = std::move(stack[start]);
      std::vector<std::unique_ptr<ExprNode>> args;
      for (size_t i = start + 1; i < stack.size(); i ++) args.push_back(std::move(stack[i]));
      auto call = std::make_unique<CallExprNode>(std::move(fn), std::move(args));
      stack.clear();
      stack.push_back(std::move(call));
      return;
    }

    std::vector<std::unique_ptr<ExprNode>> terms;
    std::vector<char> additiveOperators;
    terms.push_back(std::move(stack[start]));
    for (size_t i = start; i < operators.size(); ++i) {
      auto right = std::move(stack[i + 1]);
      if (operators[i] == '*') {
        auto left = std::move(terms.back());
        terms.back() = std::make_unique<BinopExprNode>(
            '*', std::move(left), std::move(right));
      } else {
        additiveOperators.push_back(operators[i]);
        terms.push_back(std::move(right));
      }
    }
    std::unique_ptr<ExprNode> left = std::move(terms[0]);
    for (size_t i = 0; i < additiveOperators.size(); ++i) {
      left = std::make_unique<BinopExprNode>(
          additiveOperators[i], std::move(left), std::move(terms[i + 1]));
    }
    size_t opSize = operators.size();
    for (size_t i = start; i < opSize; i ++) {
      operators.pop_back();
      stack.pop_back();
    }
    stack.pop_back();
    stack.push_back(std::move(left));
    // printf("  [reduceOpt] %d %lu %lu\n", start, operators.size(), stack.size());
  };

  std::function<std::unique_ptr<ExprNode>(size_t&)> parseExpr;

  std::function<std::unique_ptr<ExprNode>(size_t&)> parseLet = [&](size_t& start) {
    std::string var = tokens[start+1];
    if (tokens[start+2] == "=") {
      auto next = start + 3;
      auto decl = parseExpr(next);
      auto body = parseExpr(next);
      auto let = std::make_unique<LetExprNode>(var, std::move(decl), std::move(body));
      start = next;
      return let;
    }
    std::vector<std::string> args;
    size_t j = start + 2;
    for (; j < tokens.size(); j ++) {
      if (tokens[j] == "=") break;
      args.push_back(tokens[j]);
    }
    auto next = j+1;
    auto decl = parseExpr(next);
    auto lambda = std::make_unique<LambdaExprNode>(var, args, std::move(decl));
    fns.insert(var);

    auto body = parseExpr(next);
    auto let = std::make_unique<LetExprNode>(var, std::move(lambda), std::move(body));

    // TODO fns pop lambda
    start = next;
    return let;
  };

  parseExpr = [&](size_t& start) {
    std::vector<std::unique_ptr<ExprNode>> stack;
    std::vector<char> operators;
    // printf("**parseExpr** start %ld\n", start);
    size_t i = start;
    for (; i < tokens.size();) {
      auto& tok = tokens[i];
      // printf("[process tok] i: %lu tok: %s\n", i, tok.c_str());
      if (tok == ";" || tok == "in" || tok == "then" || tok == "else") {
        reduceOpt(stack, operators);
        start = i+1;
        auto top = std::move(stack[0]);
        stack.pop_back();
        return top;
      } else if (tok == "let") {
        auto let = parseLet(i);
        stack.push_back(std::move(let));
        reduceOpt(stack, operators);
        auto top = std::move(stack[0]);
        stack.pop_back();
        start = i;
        return top;
      } else if (tok == "if") {
        auto next = i + 1;
        auto cond = parseExpr(next);
        auto then = parseExpr(next);
        auto els = parseExpr(next);

        auto ifNode = std::make_unique<IfExprNode>(std::move(cond), std::move(then), std::move(els));
        stack.push_back(std::move(ifNode));
        i = next;
      } else if (tok == "[") {
        std::vector<int> values;
        ++i;
        while (i < tokens.size() && tokens[i] != "]") {
          if (tokens[i] == ",") {
            ++i;
            continue;
          }
          if (tokens[i].empty() || tokens[i][0] < '0' || tokens[i][0] > '9')
            throw std::invalid_argument("tensor literal elements must be integers");
          values.push_back(std::stoi(tokens[i++]));
        }
        if (i == tokens.size())
          throw std::invalid_argument("unterminated tensor literal");
        if (values.empty())
          throw std::invalid_argument("tensor literal cannot be empty");
        stack.push_back(std::make_unique<TensorLiteralExprNode>(std::move(values)));
        ++i;
      } else if (tok[0] >= '0' && tok[0] <= '9') {
        int v = std::stoi(tok);
        auto num = std::make_unique<NumberExprNode>(v);
        stack.push_back(std::move(num));
        i++;
      } else if (tok[0] == '+' || tok[0] == '-' || tok[0] == '*') {
        operators.push_back(tok[0]);
        i++;
      } else {
        auto var = std::make_unique<VarExprNode>(tok);
        stack.push_back(std::move(var));
        i++;
      }

      if (i == tokens.size()) {
        reduceOpt(stack, operators);
      }
    }
    auto top = std::move(stack[0]);
    stack.pop_back();
    start = tokens.size();
    return top;
  };

  size_t start = 0;
  std::unique_ptr<ExprNode> node = parseExpr(start);
  printf("===== ast =====\n");
  printf("%s\n", node->dump().c_str());
  
  return node;
}

inline bool isTensorExpressionIdentifier(std::string_view name) {
  auto letter = [](char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_';
  };
  if (name.empty() || !letter(name.front()))
    return false;
  for (char c : name)
    if (!letter(c) && !(c >= '0' && c <= '9'))
      return false;
  constexpr std::string_view reserved[] = {
      "let", "in", "if", "then", "else", "fun", "rec", "match", "with",
      "type", "module", "open", "true", "false", "print", "and", "or"};
  for (auto word : reserved)
    if (name == word)
      return false;
  return true;
}

// Validate the deliberately small parameterized-expression grammar before
// entering the legacy parser, whose general ML stack indexing is unchecked.
// Reuses that parser/AST and precedence; no second expression compiler exists.
inline std::unique_ptr<ExprNode> parseTensorExpression(std::string_view source) {
  std::string normalized;
  bool operand = true;
  size_t position = 0;
  while (position < source.size()) {
    const auto c = static_cast<unsigned char>(source[position]);
    if (std::isspace(c)) {
      ++position;
      continue;
    }
    const size_t start = position;
    if (operand) {
      if (c >= '0' && c <= '9') {
        while (position < source.size() && source[position] >= '0' &&
               source[position] <= '9')
          ++position;
        int32_t value;
        auto parsed = std::from_chars(source.data() + start,
                                       source.data() + position, value);
        if (parsed.ec != std::errc{})
          throw std::invalid_argument("tensor expression constant outside i32 range");
      } else {
        while (position < source.size()) {
          char ch = source[position];
          if (!((ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') ||
                (ch >= '0' && ch <= '9') || ch == '_'))
            break;
          ++position;
        }
        if (!isTensorExpressionIdentifier(source.substr(start, position - start)))
          throw std::invalid_argument(
              "tensor expression expected an identifier or nonnegative i32 constant");
      }
    } else {
      if (c != '+' && c != '-' && c != '*')
        throw std::invalid_argument("tensor expression expected +, - or *");
      ++position;
    }
    normalized.append(source.substr(start, position - start));
    normalized += ' ';
    operand = !operand;
  }
  if (normalized.empty() || operand)
    throw std::invalid_argument("tensor expression is empty or missing an operand");
  return parse(normalized);
}
}

#endif // SCONEML_PARSER_H
