#ifndef SCONEML_PARSER_H
#define SCONEML_PARSER_H

#include <memory>
#include <string>
#include <unordered_set>
#include <functional>
#include <memory>
#include "Ast.h"

namespace sconeml {

void tokenize(std::string& input, std::vector<std::string>& tokens) {
  std::string tok;
  for (auto c : input) {
    if (c == ' ' || c == '\n' || c == ';') {
      if (tok.size() > 0) tokens.push_back(tok);
      tok.resize(0);
    } else if (c == '(' || c == ')' || c == '{' || c == '}' || c == '+' || c == '-' || c == '*' || c == '/' || c == '^' || c == '!' || c == '~' || c == '>' || c == '<') {
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

    std::unique_ptr<ExprNode> left = std::move(stack[start]);
    for (size_t i = start; i < operators.size(); i ++) {
      auto right = std::move(stack[i+1]);
      left = std::make_unique<BinopExprNode>(operators[i], std::move(left), std::move(right));
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
      } else if (tok[0] >= '0' && tok[0] <= '9') {
        int v = std::stoi(tok);
        auto num = std::make_unique<NumberExprNode>(v);
        stack.push_back(std::move(num));
        i++;
      } else if (tok[0] == '+' || tok[0] == '-') {
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
}

#endif // SCONEML_PARSER_H