#ifndef SCONEML_UNWRAP_LET_H
#define SCONEML_UNWRAP_LET_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace sconeml {
struct UnwrapLetPass : public mlir::PassWrapper<UnwrapLetPass, mlir::OperationPass<mlir::ModuleOp>> {
  void runOnOperation() override {
    getOperation().walk([&](mlir::Operation* op) {
      if (auto letOp = mlir::dyn_cast_or_null<sconeml::letalg::LetOp>(op)) {
        auto& region = letOp.getRegion();
        std::vector<mlir::Operation*> ops;
        for (auto& innerOp : region.getOps()) {
          ops.push_back(&innerOp);
        }
        // pop last yield op
        ops.pop_back();
        for (auto* innerOp : ops) {
          innerOp->moveBefore(op);
        }

        op->replaceAllUsesWith(ops.back());
        op->erase();
      }
    });
  }
};

std::unique_ptr<mlir::Pass> createUnwrapLetPass() { return std::make_unique<UnwrapLetPass>(); }

}
#endif // SCONEML_UNWRAP_LET_H