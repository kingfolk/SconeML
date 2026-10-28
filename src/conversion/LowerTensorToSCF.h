#ifndef SCONEML_LOWER_TENSOR_TO_SCF_H
#define SCONEML_LOWER_TENSOR_TO_SCF_H

#include "src/dialect/LetAlgDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace sconeml {

class TensorMapLowering
    : public mlir::OpRewritePattern<sconeml::letalg::TensorMapOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(sconeml::letalg::TensorMapOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto inputType = mlir::dyn_cast<mlir::MemRefType>(op.getInput().getType());
    auto outputType = mlir::dyn_cast<mlir::MemRefType>(op.getOutput().getType());
    if (!inputType || !outputType || inputType.getRank() != 1 ||
        outputType.getRank() != 1 ||
        inputType.getElementType() != outputType.getElementType())
      return rewriter.notifyMatchFailure(
          op, "expected matching rank-1 memref operands");

    mlir::Block &mapBody = op.getBody().front();
    auto yield = mlir::dyn_cast<sconeml::letalg::YieldOp>(mapBody.getTerminator());
    if (!yield || mapBody.getNumArguments() != 1 ||
        mapBody.getArgument(0).getType() != inputType.getElementType())
      return rewriter.notifyMatchFailure(
          op, "expected one element-typed body argument and yield terminator");

    mlir::Location loc = op.getLoc();
    mlir::Value zero =
        mlir::arith::ConstantIndexOp::create(rewriter, loc, 0);
    mlir::Value one = mlir::arith::ConstantIndexOp::create(rewriter, loc, 1);
    mlir::Value size =
        mlir::memref::DimOp::create(rewriter, loc, op.getInput(), zero);

    mlir::scf::ForOp::create(
        rewriter,
        loc, zero, size, one, mlir::ValueRange{},
        [&](mlir::OpBuilder &builder, mlir::Location bodyLoc,
            mlir::Value index, mlir::ValueRange) {
          mlir::Value element = mlir::memref::LoadOp::create(
              builder, bodyLoc, op.getInput(), mlir::ValueRange{index});

          mlir::IRMapping mapping;
          mapping.map(mapBody.getArgument(0), element);
          for (mlir::Operation &bodyOp : mapBody.without_terminator())
            builder.clone(bodyOp, mapping);

          mlir::Value mapped = mapping.lookupOrDefault(yield.getExpr());
          mlir::memref::StoreOp::create(
              builder, bodyLoc, mapped, op.getOutput(), mlir::ValueRange{index});
          mlir::scf::YieldOp::create(builder, bodyLoc);
        });

    rewriter.eraseOp(op);
    return mlir::success();
  }
};

struct LowerTensorToSCFPass
    : public mlir::PassWrapper<LowerTensorToSCFPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerTensorToSCFPass)

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::arith::ArithDialect, mlir::memref::MemRefDialect,
                    mlir::scf::SCFDialect>();
  }

  void runOnOperation() override {
    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<TensorMapLowering>(&getContext());
    if (mlir::failed(mlir::applyPatternsGreedily(getOperation(),
                                                 std::move(patterns))))
      signalPassFailure();
  }
};

inline std::unique_ptr<mlir::Pass> createLowerTensorToSCFPass() {
  return std::make_unique<LowerTensorToSCFPass>();
}

} // namespace sconeml

#endif
