#ifndef CAKEML_PARSER_H
#define CAKEML_PARSER_H

#include <memory>
#include <string>
#include <unordered_set>
#include <functional>
#include <memory>
#include "Ast.h"

namespace cakeml {

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
    printf("%s\n", t.c_str());
  }

  std::vector<std::unique_ptr<ExprNode>> stack;
  std::vector<char> operators;
  std::unordered_set<std::string> fns;
  auto reduceOpt = [&](int start) {
    // printf("[reduceOpt] %d %lu %lu\n", start, operators.size(), stack.size());
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
    for (size_t i = start; i < operators.size(); i ++) {
      operators.pop_back();
      stack.pop_back();
    }
    stack.pop_back();
    stack.push_back(std::move(left));
  };

  std::function<size_t(size_t)> parseExpr = [&](size_t start) {
    size_t stackPos = stack.size();
    size_t i = start;
    for (; i < tokens.size();) {
      auto& tok = tokens[i];
      // printf("[process tok] i: %lu tok: %s\n", i, tok.c_str());
      if (tok == ";" || tok == "in" || tok == "then" || tok == "else") {
        reduceOpt(stackPos);
        i++;
        return i;
      } else if (tok == "let") {
        std::string var = tokens[i+1];
        if (tokens[i+2] == "=") {
          auto next = i + 3;
          next = parseExpr(next);
          auto decl = std::move(stack.back());
          stack.pop_back();
          next = parseExpr(next);
          auto body = std::move(stack.back());
          stack.pop_back();
          auto let = std::make_unique<LetExprNode>(var, std::move(decl), std::move(body));
          stack.push_back(std::move(let));
          return next;
        } else {
          std::vector<std::string> args;
          size_t j = i + 2;
          for (; j < tokens.size(); j ++) {
            if (tokens[j] == "=") break;
            args.push_back(tokens[j]);
          }
          auto next = parseExpr(j+1);
          auto decl = std::move(stack.back());
          stack.pop_back();
          auto lambda = std::make_unique<LambdaExprNode>(var, args, std::move(decl));
          fns.insert(var);

          next = parseExpr(next);
          auto body = std::move(stack.back());
          stack.pop_back();
          auto let = std::make_unique<LetExprNode>(var, std::move(lambda), std::move(body));
          stack.push_back(std::move(let));

          // TODO fns pop lambda
          return next;
        }
      } else if (tok == "if") {
        auto next = i + 1;
        next = parseExpr(next);
        auto cond = std::move(stack.back());
        stack.pop_back();
        next = parseExpr(next);
        auto then = std::move(stack.back());
        stack.pop_back();
        next = parseExpr(next);
        auto els = std::move(stack.back());
        stack.pop_back();

        auto ifNode = std::make_unique<IfExprNode>(std::move(cond), std::move(then), std::move(els));
        stack.push_back(std::move(ifNode));
        return next;
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
        reduceOpt(stackPos);
      }
    }
    return i;
  };

  parseExpr(0);
  printf("===== ast =====\n");
  printf("%s\n", stack.front()->dump().c_str());
  
  return std::move(stack.front());
}
}

#endif // CAKEML_PARSER_H