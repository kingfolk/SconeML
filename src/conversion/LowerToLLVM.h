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
mlir::ConversionPatternRewriter &rewriter, mlir::IRMapping& valueMapping) {
  auto& sourceRegion = sourceOp->getRegion(0);
  auto& sourceBlock = sourceRegion.front();
  auto& targetBlock = targetFunc.getBody().front();
  
  rewriter.setInsertionPointToStart(&targetBlock);

  llvm::SmallVector<mlir::Operation*> opsToErase;
  for (auto& op : sourceBlock) {
    auto* clonedOp = rewriter.clone(op, valueMapping);
    for (auto [originalResult, clonedResult] : 
        llvm::zip(op.getResults(), clonedOp->getResults())) {
      valueMapping.map(originalResult, clonedResult);
    }

    opsToErase.push_back(&op);
  }

  for (auto* op : opsToErase) {
    rewriter.eraseOp(op);
  }
}

class LambdaLowering : public mlir::OpConversionPattern<sconeml::letalg::LambdaOp> {
public:
  using mlir::OpConversionPattern<sconeml::letalg::LambdaOp>::OpConversionPattern;
  
  mlir::LogicalResult matchAndRewrite(
  sconeml::letalg::LambdaOp op, 
  sconeml::letalg::LambdaOp::Adaptor adaptor, 
  mlir::ConversionPatternRewriter& rewriter) const override {
    auto rootFunc = op->getParentOfType<mlir::func::FuncOp>();
    mlir::Location loc = op.getLoc();
    std::string fnName = op.getName().str();
    auto opType = mlir::dyn_cast_or_null<mlir::FunctionType>(op.getType());

    if (!opType) return mlir::failure();

    llvm::SmallVector<sconeml::letalg::ApplyOp> applyUsers;
    for (auto user : op->getUsers()) {
      // TODO currently all users are apply. this is not true in future as function is first class citizen.
      if (auto applyOp = mlir::dyn_cast<sconeml::letalg::ApplyOp>(user)) {
        applyUsers.push_back(applyOp);
      }
    }
    
    auto module = op->getParentOfType<mlir::ModuleOp>();
    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(module.getBody(), module.getBody()->end());
    
    auto fn = rewriter.create<mlir::func::FuncOp>(loc, fnName, opType);
    fn.setPrivate();
    fn.addEntryBlock();
    
    mlir::IRMapping mapping;
    for (auto [sourceArg, targetArg] : llvm::zip(op.getRegion().getArguments(), fn.getArguments())) {
      mapping.map(sourceArg, targetArg);
    }

    moveOperationsToFunc(op.getOperation(), fn, rewriter, mapping);

    for (auto applyOp : applyUsers) {
      if (!applyOp.getOperation() || applyOp.getOperation()->getParentOp() == nullptr) {
        continue;
      }

      rewriter.setInsertionPoint(applyOp);
      auto callOp = rewriter.create<mlir::func::CallOp>(
        applyOp.getLoc(),
        fn,
        applyOp.getVars()
      );

      rewriter.replaceOp(applyOp, callOp.getResults());
    }

    rewriter.eraseOp(op);
    return mlir::success();
  }
};


class YieldLowering : public mlir::OpConversionPattern<sconeml::letalg::YieldOp> {
  public:
  using mlir::OpConversionPattern<sconeml::letalg::YieldOp>::OpConversionPattern;
  
  mlir::LogicalResult matchAndRewrite(sconeml::letalg::YieldOp op, 
  sconeml::letalg::YieldOp::Adaptor adaptor, 
  mlir::ConversionPatternRewriter& rewriter) const override {
    mlir::Location loc = op.getLoc();
    rewriter.setInsertionPoint(op);
    auto retOp = rewriter.create<mlir::func::ReturnOp>(loc, op.getExpr());
    rewriter.replaceOp(op, retOp);

    return mlir::success();
  }
};

struct LowerToLLVMLoweringPass
  : public mlir::PassWrapper<LowerToLLVMLoweringPass, mlir::OperationPass<>> {

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::LLVM::LLVMDialect>();
  }
  void runOnOperation() final {
    mlir::Operation* op = getOperation();
    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<LambdaLowering>(patterns.getContext());
    patterns.add<YieldLowering>(patterns.getContext());

    mlir::LLVMConversionTarget target(getContext());
    target.addIllegalDialect<sconeml::letalg::LetAlgDialect>();
    target.addLegalDialect<mlir::BuiltinDialect>();
    target.addLegalDialect<mlir::func::FuncDialect>();
    target.addLegalDialect<mlir::arith::ArithDialect>();
    target.addLegalDialect<mlir::LLVM::LLVMDialect>();

    if (mlir::failed(mlir::applyPartialConversion(op, target, std::move(patterns))))
      mlir::Pass::signalPassFailure();
  }
};

std::unique_ptr<mlir::Pass> createLowerToLLVMPass() {
  return std::make_unique<LowerToLLVMLoweringPass>();
}

}

#endif // LOWER_TO_LLVM_H