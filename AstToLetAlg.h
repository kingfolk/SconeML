#ifndef CAKEML_AST_TO_LETALG_H
#define CAKEML_AST_TO_LETALG_H

#include "Ast.h"
#include "LetAlgDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#include <string>
#include <vector>

namespace cakeml {
struct LetContext {
  std::vector<std::tuple<std::string, int>> variables;
  std::vector<mlir::Value> values;
  mlir::Region* region;

  void push(std::string name, mlir::Value arg) {
    variables.push_back(std::make_tuple(name, variables.size()));
    values.push_back(arg);
  }

  int find(std::string name) {
    for (size_t i = 0; i < variables.size(); i ++) {
      if (std::get<0>(variables[i]) == name) {
        return i;
      }
    }
    return -1;
  }

  mlir::Value getArgument(size_t index) {
    return region->getArgument(index);
  }
};

mlir::Value translateExpr(mlir::OpBuilder& builder, ExprNode* expr, LetContext& ctx);

mlir::Value translateLet(mlir::OpBuilder& builder, LetExprNode* let) {
  auto loc = builder.getUnknownLoc();
  LetContext ctx;
  cakeml::ExprNode *finalBody;
  std::function<void(LetExprNode*)> processLet = [&](LetExprNode* letNode) {
    auto name = letNode->getVar();
    ctx.push(name, translateExpr(builder, letNode->getDecl(), ctx));

    auto body = letNode->getBody();
    if (body->getKind() == ExprNode::Kind_Let) {
      processLet(reinterpret_cast<LetExprNode*>(body));
    } else {
      finalBody = body;
    }
  };
  processLet(let);

  auto letOp = builder.create<mlir::letalg::LetOp>(loc, builder.getI32Type(), ctx.values);
  mlir::Region& region = letOp.getRegion();
  mlir::Block* scopeBlock = builder.createBlock(&region);
  std::vector<mlir::Type> blockArgTps;
  for (auto& arg : ctx.values) {
    blockArgTps.push_back(arg.getType());
  }
  region.addArguments(blockArgTps,  std::vector<mlir::Location>(blockArgTps.size(), loc));
  ctx.region = &region;
  builder.setInsertionPointToStart(scopeBlock);
  auto v = translateExpr(builder, finalBody, ctx);
  builder.create<mlir::letalg::YieldOp>(loc, v.getType(), v);
  letOp.getResult().setType(v.getType());

  builder.setInsertionPointAfter(letOp);
  return letOp;
}

mlir::Value translateLambda(mlir::OpBuilder& builder, LambdaExprNode* lambda) {
  auto loc = builder.getUnknownLoc();

  std::vector<mlir::Value> vals{};
  auto lambdaOp = builder.create<mlir::letalg::LambdaOp>(loc, builder.getI32Type(), lambda->getFn(), vals);
  mlir::Region& region = lambdaOp.getRegion();
  mlir::Block* scopeBlock = builder.createBlock(&region);

  LetContext ctx;
  std::vector<mlir::Type> blockArgTps;
  for (auto& arg : lambda->getArgs()) {
    ctx.variables.push_back(std::make_tuple(arg, ctx.variables.size()));
    blockArgTps.push_back(builder.getI32Type());
  }
  region.addArguments(blockArgTps, std::vector<mlir::Location>(blockArgTps.size(), loc));
  ctx.region = &region;
  builder.setInsertionPointToStart(scopeBlock);
  auto v = translateExpr(builder, lambda->getBody(), ctx);
  builder.create<mlir::letalg::YieldOp>(loc, v.getType(), v);
  lambdaOp.getResult().setType(
    mlir::FunctionType::get(builder.getContext(), blockArgTps, mlir::TypeRange({v.getType()}))
  );

  builder.setInsertionPointAfter(lambdaOp);
  return lambdaOp;
}

mlir::Value translateExpr(mlir::OpBuilder& builder, ExprNode* expr, LetContext& ctx) {
  auto loc = builder.getUnknownLoc();
  auto kind = expr->getKind();
  if (kind == ExprNode::Kind_Let) {
    auto let = reinterpret_cast<LetExprNode*>(expr);
    return translateLet(builder, let);
  } else if (kind == ExprNode::Kind_Lambda) {
    auto lambda = reinterpret_cast<LambdaExprNode*>(expr);
    return translateLambda(builder, lambda);
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
    return builder.create<mlir::letalg::ApplyOp>(loc, returnTp, fn, args);
  } else if (kind == ExprNode::Kind_If) {
    auto ifNode = reinterpret_cast<IfExprNode*>(expr);
    return builder.create<mlir::scf::IfOp>(loc, translateExpr(builder, ifNode->getCond(), ctx),
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
    auto idx = ctx.find(var->getName());
    if (idx == -1) {
      throw std::invalid_argument("variable not found " + expr->dump());
    }
    return ctx.getArgument(idx);
  } else if (kind == ExprNode::Kind_Num) {
    auto num = reinterpret_cast<NumberExprNode*>(expr);
    mlir::IntegerAttr i32Attr = builder.getIntegerAttr(builder.getI32Type(), num->getValue());
    return builder.create<mlir::arith::ConstantOp>(loc, i32Attr);
  } else if (kind == ExprNode::Kind_BinOp) {
    auto binop = reinterpret_cast<BinopExprNode*>(expr);
    auto l = translateExpr(builder, binop->getL(), ctx);
    auto r = translateExpr(builder, binop->getR(), ctx);
    if (binop->getOp() == '-') {
      return builder.create<mlir::arith::SubIOp>(loc, l, r);
    }
    return builder.create<mlir::arith::AddIOp>(loc, l, r);
  } else {
    throw std::invalid_argument("unsupported expr to translate " + expr->dump());
  }
}

mlir::Value translate(mlir::OpBuilder& builder, ExprNode* expr) {
  LetContext ctx;
  return translateExpr(builder, expr, ctx);
}
}

#endif // CAKEML_AST_TO_LETALG_H