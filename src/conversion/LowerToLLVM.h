#ifndef LOWER_TO_LLVM_H
#define LOWER_TO_LLVM_H

#include "src/dialect/LetAlgDialect.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"

namespace sconeml {

void moveOperationsToFunc(mlir::Operation *sourceOp, mlir::func::FuncOp targetFunc, 
mlir::OpBuilder &builder, mlir::IRMapping& valueMapping) {
  auto& sourceRegion = sourceOp->getRegion(0);
  auto& sourceBlock = sourceRegion.front();
  auto& targetBlock = targetFunc.getBody().front();
  
  builder.setInsertionPoint(targetBlock.getTerminator());
  llvm::SmallVector<mlir::Operation*> opsToErase;
  for (auto& op : sourceBlock) {
    auto yieldOp = mlir::dyn_cast_or_null<sconeml::letalg::YieldOp>(op);
    if (yieldOp) {
      auto ret = valueMapping.lookup(yieldOp.getExpr());
      builder.create<mlir::func::ReturnOp>(op.getLoc(), ret);
    } else {
      auto* clonedOp = builder.clone(op, valueMapping);

      for (auto [originalResult, clonedResult] : 
          llvm::zip(op.getResults(), clonedOp->getResults())) {
        valueMapping.map(originalResult, clonedResult);
      }
    }

    opsToErase.push_back(&op);
  }

  for (auto* op : opsToErase) {
    op->erase();
  }
}

class LambdaLowering : public mlir::OpConversionPattern<sconeml::letalg::LambdaOp> {
  public:
  using mlir::OpConversionPattern<sconeml::letalg::LambdaOp>::OpConversionPattern;
  
  mlir::LogicalResult matchAndRewrite(sconeml::letalg::LambdaOp op, 
  sconeml::letalg::LambdaOp::Adaptor adaptor, 
  mlir::ConversionPatternRewriter& rewriter) const override {
    printf("<<< LambdaLowering\n");
    auto rootFunc = op->getParentOfType<mlir::func::FuncOp>();
    mlir::Location loc = op.getLoc();

    std::string fnName = op.getName().str();
    auto opType = mlir::dyn_cast_or_null<mlir::FunctionType>(op.getType());
    if (!opType) return mlir::failure();

    rewriter.setInsertionPoint(rootFunc);
    // mlir::StringAttr stringAttr = rewriter.getStringAttr(fnName);
    // mlir::Value token = rewriter.create<mlir::arith::ConstantOp>(loc, stringAttr);
    auto fn = rewriter.create<mlir::func::FuncOp>(loc, fnName, opType);
    fn.setPrivate();
    fn.addEntryBlock();

    printf("<<< LambdaLowering1\n");

    std::vector<mlir::Operation*> users;
    for (auto user : op->getUsers()) {
      users.push_back(user);
    }

    mlir::IRMapping mapping;
    for (auto [sourceArg, targetArg] : llvm::zip(op.getRegion().getArguments(), fn.getArguments())) {
      mapping.map(sourceArg, targetArg);
    }

    printf("<<< LambdaLowering2 %lu\n", users.size());

    moveOperationsToFunc(op.getOperation(), fn, rewriter, mapping);

    printf("<<< LambdaLowering3\n");

    // TODO every user a token to call
    printf("~~ users\n");
    for (auto* user : users) {
      printf("  ~~ user\n");
      rewriter.setInsertionPoint(user);
      mlir::StringAttr stringAttr = rewriter.getStringAttr(fnName);
      mlir::Value token = rewriter.create<mlir::arith::ConstantOp>(loc, stringAttr);
      // rewriter.setInsertionPoint(rootFunc);
      user->replaceUsesOfWith(op.getResult(), token);
      // for (OpResult result : op->getResults()) {
      //   user->replaceUsesOfWith(result, token);
      // }
    }
    printf("  ~~ users done\n");
    rootFunc.dump();
    // op.replaceAllUsesWith(token);
    auto parent = rootFunc.getOperation()->getParentOp();
    parent->dump();
    op.erase();

    // auto parent = rootFunc.getOperation()->getParentOp();
    // parent->dump();

    return mlir::success();
  }
};

class ApplyLowering : public mlir::OpConversionPattern<sconeml::letalg::ApplyOp> {
  public:
  using mlir::OpConversionPattern<sconeml::letalg::ApplyOp>::OpConversionPattern;
  
  mlir::LogicalResult matchAndRewrite(sconeml::letalg::ApplyOp op, 
  sconeml::letalg::ApplyOp::Adaptor adaptor, 
  mlir::ConversionPatternRewriter& rewriter) const override {
    printf("<<< ApplyLowering\n");
    auto rootModule = op->getParentOfType<mlir::ModuleOp>();
    auto symbol = rootModule.lookupSymbol("f");
    if (symbol) {
      symbol->dump();
    }

    return mlir::success();
  }
};

class YieldLowering : public mlir::OpConversionPattern<sconeml::letalg::YieldOp> {
  public:
  using mlir::OpConversionPattern<sconeml::letalg::YieldOp>::OpConversionPattern;
  
  mlir::LogicalResult matchAndRewrite(sconeml::letalg::YieldOp op, 
  sconeml::letalg::YieldOp::Adaptor adaptor, 
  mlir::ConversionPatternRewriter& rewriter) const override {
    printf("<<< YieldLowering\n");
    mlir::Location loc = op.getLoc();
    rewriter.setInsertionPoint(op);
    rewriter.create<mlir::func::ReturnOp>(loc, op.getExpr());
    auto rootModule = op->getParentOfType<mlir::ModuleOp>();
    op.erase();

    rootModule.dump();

    return mlir::success();
  }
};

struct LowerToLLVMLoweringPass
  : public mlir::PassWrapper<LowerToLLVMLoweringPass, mlir::OperationPass<>> {
  
  void getDependentDialects(mlir::DialectRegistry& registry) const override {
    registry.insert<mlir::LLVM::LLVMDialect>();
  }
  void runOnOperation() final {
    mlir::Operation* op = getOperation();
    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<YieldLowering>(patterns.getContext());
    patterns.add<LambdaLowering>(patterns.getContext());
    patterns.add<ApplyLowering>(patterns.getContext());

    mlir::LLVMConversionTarget target(getContext());
    target.addIllegalOp<sconeml::letalg::YieldOp>();
    target.addIllegalOp<sconeml::letalg::LambdaOp>();
    target.addIllegalOp<sconeml::letalg::ApplyOp>();
    // target.addIllegalDialect<sconeml::letalg::LetAlgDialect>();
    target.addLegalDialect<mlir::LLVM::LLVMDialect>();
    target.addLegalDialect<mlir::func::FuncDialect>();
    if (mlir::failed(mlir::applyPartialConversion(op, target, std::move(patterns))))
      mlir::Pass::signalPassFailure();
  }
};

std::unique_ptr<mlir::Pass> createLowerToLLVMPass() {
  return std::make_unique<LowerToLLVMLoweringPass>();
}

}

#endif // LOWER_TO_LLVM_H