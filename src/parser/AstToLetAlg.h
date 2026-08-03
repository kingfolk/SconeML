#ifndef SCONEML_AST_TO_LETALG_H
#define SCONEML_AST_TO_LETALG_H

#include "Ast.h"
#include "src/dialect/LetAlgDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#include <string>
#include <vector>

namespace sconeml {
struct TranslateContext {
  std::vector<std::tuple<std::string, int>> variables;
  std::vector<mlir::Value> values;
  mlir::Region* region;
  TranslateContext* parent;

  void push(std::string name, mlir::Value arg) {
    variables.push_back(std::make_tuple(name, variables.size()));
    values.push_back(arg);
  }

  mlir::Value find(std::string name) {
    if (values.size() > 0) {
      for (int i = variables.size() - 1; i >= 0; i --) {
        if (std::get<0>(variables[i]) == name) {
          return values[i];
        }
      }
      return nullptr;
    }

    if (!region) throw std::invalid_argument("region not set in translate context");

    for (size_t i = 0; i < variables.size(); i ++) {
      if (std::get<0>(variables[i]) == name) {
        return region->getArgument(i);
      }
    }
    if (!parent) return nullptr;
    return parent->find(name);
  }
};

mlir::Value translateExpr(mlir::OpBuilder& builder, ExprNode* expr, TranslateContext& ctx);

mlir::Value translateTensorLiteral(mlir::OpBuilder &builder,
                                   TensorLiteralExprNode *literal) {
  auto loc = builder.getUnknownLoc();
  auto type = mlir::MemRefType::get(
      {static_cast<int64_t>(literal->getValues().size())},
      builder.getI32Type());
  auto buffer = mlir::memref::AllocOp::create(builder, loc, type);
  for (size_t i = 0; i < literal->getValues().size(); ++i) {
    auto index = mlir::arith::ConstantIndexOp::create(builder, loc, i);
    auto value = mlir::arith::ConstantOp::create(
        builder, loc, builder.getI32IntegerAttr(literal->getValues()[i]));
    mlir::memref::StoreOp::create(builder, loc, value, buffer.getResult(),
                                  mlir::ValueRange{index.getResult()});
  }
  return buffer.getResult();
}

mlir::Value translateTensorElementExpr(mlir::OpBuilder &builder, ExprNode *expr,
                                       const std::string &tensorName,
                                       mlir::Value element,
                                       TranslateContext &ctx) {
  auto loc = builder.getUnknownLoc();
  if (expr->getKind() == ExprNode::Kind_Var) {
    auto *variable = static_cast<VarExprNode *>(expr);
    if (variable->getName() == tensorName)
      return element;
    auto value = ctx.find(variable->getName());
    if (!value || mlir::isa<mlir::ShapedType>(value.getType()))
      throw std::invalid_argument("unsupported tensor expression variable " +
                                  variable->getName());
    return value;
  }
  if (expr->getKind() == ExprNode::Kind_Num) {
    auto *number = static_cast<NumberExprNode *>(expr);
    return mlir::arith::ConstantOp::create(
        builder, loc, builder.getI32IntegerAttr(number->getValue()));
  }
  if (expr->getKind() == ExprNode::Kind_BinOp) {
    auto *binary = static_cast<BinopExprNode *>(expr);
    auto left = translateTensorElementExpr(builder, binary->getL(), tensorName,
                                           element, ctx);
    auto right = translateTensorElementExpr(builder, binary->getR(), tensorName,
                                            element, ctx);
    if (binary->getOp() == '*')
      return mlir::arith::MulIOp::create(builder, loc, left, right);
    if (binary->getOp() == '-')
      return mlir::arith::SubIOp::create(builder, loc, left, right);
    return mlir::arith::AddIOp::create(builder, loc, left, right);
  }
  throw std::invalid_argument("unsupported tensor expression " + expr->dump());
}

mlir::Value translateTensorMap(mlir::OpBuilder &builder, ExprNode *expr,
                               const std::string &tensorName, mlir::Value input,
                               TranslateContext &ctx) {
  auto loc = builder.getUnknownLoc();
  auto inputType = mlir::dyn_cast<mlir::MemRefType>(input.getType());
  if (!inputType || inputType.getRank() != 1)
    throw std::invalid_argument("tensor map input must be a rank-1 memref");
  auto output = mlir::memref::AllocOp::create(builder, loc, inputType);
  auto map = sconeml::letalg::TensorMapOp::create(
      builder, loc, input, output.getResult());
  auto *body = builder.createBlock(&map.getBody(), {},
                                   {inputType.getElementType()}, {loc});
  builder.setInsertionPointToStart(body);
  auto result = translateTensorElementExpr(
      builder, expr, tensorName, body->getArgument(0), ctx);
  sconeml::letalg::YieldOp::create(builder, loc, result);
  builder.setInsertionPointAfter(map);
  return output.getResult();
}

mlir::Value translateLet(mlir::OpBuilder& builder, LetExprNode* let, TranslateContext& parent) {
  auto loc = builder.getUnknownLoc();
  TranslateContext ctx;
  ctx.parent = &parent;
  sconeml::ExprNode *finalBody;
  auto letOp = builder.create<sconeml::letalg::LetOp>(loc, builder.getI32Type(), 0, mlir::ValueRange{});
  mlir::Region& region = letOp.getRegion();
  mlir::Block* scopeBlock = builder.createBlock(&region);
  ctx.region = &region;

  builder.setInsertionPointToStart(scopeBlock);
  if (let->getDecl()->getKind() == ExprNode::Kind_TensorLiteral) {
    auto input = translateExpr(builder, let->getDecl(), ctx);
    ctx.push(let->getVar(), input);
    auto output = translateTensorMap(builder, let->getBody(), let->getVar(),
                                     input, ctx);
    letOp.setDeclCnt(1);
    builder.create<sconeml::letalg::YieldOp>(loc, output);
    letOp.getResult().setType(output.getType());
    builder.setInsertionPointAfter(letOp);
    return letOp;
  }

  std::function<void(LetExprNode*)> processLet = [&](LetExprNode* letNode) {
    auto name = letNode->getVar();
    auto arg = translateExpr(builder, letNode->getDecl(), ctx);
    ctx.push(name, arg);

    auto body = letNode->getBody();
    if (body->getKind() == ExprNode::Kind_Let) {
      processLet(reinterpret_cast<LetExprNode*>(body));
    } else {
      finalBody = body;
    }
  };
  processLet(let);
  letOp.setDeclCnt(ctx.values.size());
  auto v = translateExpr(builder, finalBody, ctx);
  builder.create<sconeml::letalg::YieldOp>(loc, v);
  letOp.getResult().setType(v.getType());

  builder.setInsertionPointAfter(letOp);
  return letOp;
}

mlir::Value translateLambda(mlir::OpBuilder& builder, LambdaExprNode* lambda, TranslateContext& parent) {
  auto loc = builder.getUnknownLoc();

  std::vector<mlir::Value> vals{};
  auto lambdaOp = builder.create<sconeml::letalg::LambdaOp>(loc, builder.getI32Type(), lambda->getFn(), vals);
  mlir::Region& region = lambdaOp.getRegion();
  mlir::Block* scopeBlock = builder.createBlock(&region);

  TranslateContext ctx;
  ctx.parent = &parent;
  std::vector<mlir::Type> blockArgTps;
  for (auto& arg : lambda->getArgs()) {
    ctx.variables.push_back(std::make_tuple(arg, ctx.variables.size()));
    blockArgTps.push_back(builder.getI32Type());
  }
  region.addArguments(blockArgTps, std::vector<mlir::Location>(blockArgTps.size(), loc));
  ctx.region = &region;
  builder.setInsertionPointToStart(scopeBlock);
  auto v = translateExpr(builder, lambda->getBody(), ctx);
  builder.create<sconeml::letalg::YieldOp>(loc, v);
  lambdaOp.getResult().setType(
    mlir::FunctionType::get(builder.getContext(), blockArgTps, mlir::TypeRange({v.getType()}))
  );

  builder.setInsertionPointAfter(lambdaOp);
  return lambdaOp;
}

mlir::Value translateExpr(mlir::OpBuilder& builder, ExprNode* expr, TranslateContext& ctx) {
  auto loc = builder.getUnknownLoc();
  auto kind = expr->getKind();
  if (kind == ExprNode::Kind_Let) {
    auto let = reinterpret_cast<LetExprNode*>(expr);
    return translateLet(builder, let, ctx);
  } else if (kind == ExprNode::Kind_Lambda) {
    auto lambda = reinterpret_cast<LambdaExprNode*>(expr);
    return translateLambda(builder, lambda, ctx);
  } else if (kind == ExprNode::Kind_Call) {
    auto call = reinterpret_cast<CallExprNode*>(expr);
    auto fn = translateExpr(builder, call->getFn(), ctx);
    std::vector<mlir::Value> args;
    for (size_t i = 0; i < call->getArgCount(); i ++) {
      args.push_back(translateExpr(builder, call->getArg(i), ctx));
    }
    // TODO return type. return could be function
    auto ft = mlir::dyn_cast_or_null<mlir::FunctionType>(fn.getType());
    if (!ft) {
      throw std::invalid_argument("apply fn is not function type: " + call->dump());
    }
    auto returnTp = ft.getResult(0);
    // TODO type check
    if (args.size() != ft.getInputs().size()) {
      std::vector<mlir::Type> restArgs;
      for (size_t i = args.size(); i < ft.getInputs().size(); i ++) {
        restArgs.push_back(ft.getInput(i));
      }
      returnTp = mlir::FunctionType::get(builder.getContext(), restArgs, mlir::TypeRange({returnTp}));
    }
    return builder.create<sconeml::letalg::ApplyOp>(loc, returnTp, fn, args);
  } else if (kind == ExprNode::Kind_If) {
    auto ifNode = reinterpret_cast<IfExprNode*>(expr);
    auto cond = translateExpr(builder, ifNode->getCond(), ctx);
    if (cond.getType() != builder.getI1Type()) {
      mlir::IntegerAttr zeroAttr = builder.getIntegerAttr(builder.getI32Type(), 0);
      cond = builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::eq, cond, builder.create<mlir::arith::ConstantOp>(loc, zeroAttr));
    }
    return builder.create<mlir::scf::IfOp>(loc, cond,
      [&](mlir::OpBuilder& builder, mlir::Location loc) {
        auto v = translateExpr(builder, ifNode->getThen(), ctx);
        builder.create<mlir::scf::YieldOp>(loc, v);
      }, [&](mlir::OpBuilder& builder, mlir::Location loc) {
        auto v = translateExpr(builder, ifNode->getEls(), ctx);
        builder.create<mlir::scf::YieldOp>(loc, v);
      }
    ).getResult(0);
  } else if (kind == ExprNode::Kind_Var) {
    auto var = reinterpret_cast<VarExprNode*>(expr);
    auto found = ctx.find(var->getName());
    if (!found) {
      throw std::invalid_argument("variable not found " + expr->dump());
    }
    return found;
  } else if (kind == ExprNode::Kind_Num) {
    auto num = reinterpret_cast<NumberExprNode*>(expr);
    mlir::IntegerAttr i32Attr = builder.getIntegerAttr(builder.getI32Type(), num->getValue());
    return builder.create<mlir::arith::ConstantOp>(loc, i32Attr);
  } else if (kind == ExprNode::Kind_TensorLiteral) {
    return translateTensorLiteral(
        builder, static_cast<TensorLiteralExprNode *>(expr));
  } else if (kind == ExprNode::Kind_BinOp) {
    auto binop = reinterpret_cast<BinopExprNode*>(expr);
    auto l = translateExpr(builder, binop->getL(), ctx);
    auto r = translateExpr(builder, binop->getR(), ctx);
    if (binop->getOp() == '-') {
      return builder.create<mlir::arith::SubIOp>(loc, l, r);
    }
    if (binop->getOp() == '*') {
      return builder.create<mlir::arith::MulIOp>(loc, l, r);
    }
    return builder.create<mlir::arith::AddIOp>(loc, l, r);
  } else {
    throw std::invalid_argument("unsupported expr to translate " + expr->dump());
  }
}

mlir::Value translate(mlir::OpBuilder& builder, ExprNode* expr) {
  TranslateContext ctx;
  return translateExpr(builder, expr, ctx);
}
}

#endif // SCONEML_AST_TO_LETALG_H
