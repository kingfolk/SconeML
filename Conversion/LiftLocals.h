#ifndef SCONEML_LIFT_LOCALS_H
#define SCONEML_LIFT_LOCALS_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace sconeml {
struct LiftLocalsPass : public mlir::PassWrapper<LiftLocalsPass, mlir::OperationPass<mlir::ModuleOp>> {
  virtual llvm::StringRef getArgument() const override { return "lift-locals"; }

  void runOnOperation() override {
    mlir::OpBuilder builder(getOperation().getContext());
    getOperation().walk<mlir::WalkOrder::PreOrder>([&](mlir::Operation* op) {
      if (auto letOp = mlir::dyn_cast_or_null<sconeml::letalg::LetOp>(op)) {
        auto& region = letOp.getRegion();
        std::vector<mlir::Operation*> ops;
        std::vector<mlir::Value> inputs;
        int cnt = 0;
        for (auto& op : region.getOps()) {
          if (cnt == letOp.getDeclCnt()) break;
          ops.push_back(&op);
          cnt ++;
        }
        for (auto& op : ops) {
          op->moveBefore(letOp);
          auto def = op->getResult(0);
          inputs.push_back(def);
          auto& region = letOp->getRegion(0);
          region.addArgument(op->getResult(0).getType(), letOp->getLoc()); 
          def.replaceAllUsesWith(region.getArgument(region.getNumArguments()-1));
        }
        letOp->insertOperands(0, inputs);
      }
    });
  }
};

std::unique_ptr<mlir::Pass> createLiftLocalsPass() { return std::make_unique<LiftLocalsPass>(); }

}
#endif // SCONEML_LIFT_LOCALS_H
