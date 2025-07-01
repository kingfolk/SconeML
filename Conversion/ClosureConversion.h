#ifndef CAKEML_CLOSURE_CONVERSION_H
#define CAKEML_CLOSURE_CONVERSION_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace cakeml {
struct ClosureConversionPass : public mlir::PassWrapper<ClosureConversionPass, mlir::OperationPass<mlir::ModuleOp>> {
  virtual llvm::StringRef getArgument() const override { return "closure-conversion"; }

  struct Closure {
    mlir::Operation* op;
    mlir::Region* region;
  };

  void runOnOperation() override {
    std::vector<Closure> closures;

    auto popClosure = [&](mlir::Region* cur) {
      for (int i = closures.size() - 1; i >= 0; i --) {
        if (closures.size() > 0 && cur != closures.back().region) closures.pop_back();
      }
    };

    auto searchClosure = [&](mlir::Region* rg) {
      for (auto& c : closures) {
        if (c.region == rg) return &c;
      }
      return (Closure*)nullptr;
    };

    std::function<mlir::Value(mlir::Value)> bindUse = [&](mlir::Value use) {
      std::function<void(mlir::Operation*)> bind = [&](mlir::Operation* op) {
        auto& region = op->getRegion(0);
        region.insertArgument((unsigned int)0, use.getType(), op->getLoc());
        for (auto app : op->getUsers()) {
          auto applyOp = mlir::dyn_cast_or_null<mlir::letalg::ApplyOp>(app);
          // TODO consider let
          if (!applyOp) throw std::invalid_argument("user of lambda is not apply op");
          if (use.getParentRegion() == app->getParentRegion()) {
            applyOp->insertOperands(1, mlir::ValueRange{use});
          } else {
            auto region = applyOp->getParentRegion();
            auto* c = searchClosure(region);
            bind(c->op);
            applyOp->insertOperands(1, mlir::ValueRange{region->getArgument(0)});
          }
        }
      };

      if (closures.size() > 0 && use.getParentRegion() != closures.back().region) {
        bind(closures.back().op);
        return (mlir::Value)closures.back().region->getArgument(0);
      }
      return (mlir::Value)nullptr;
    };

    mlir::OpBuilder builder(getOperation().getContext());

    getOperation().walk<mlir::WalkOrder::PreOrder>([&](mlir::Operation* op) {
      // printf("  !!! op\n");
      // op->dump();
      if (auto letOp = mlir::dyn_cast_or_null<mlir::letalg::LetOp>(op)) {
        closures.push_back(Closure{
          op: op,
          region: &letOp.getRegion()
        });
      } else if (auto lambdaOp = mlir::dyn_cast_or_null<mlir::letalg::LambdaOp>(op)) {
        closures.push_back(Closure{
          op: op,
          region: &lambdaOp.getRegion()
        });
      } else {
        popClosure(op->getParentRegion());
        std::vector<mlir::Value> newOpnds;
        bool hasReplaced = false;
        for (auto arg : op->getOperands()) {
          if (auto replaced = bindUse(arg)) {
            newOpnds.push_back(replaced);
            hasReplaced = true;
          } else {
            newOpnds.push_back(arg);
          }
        }
        if (hasReplaced) {
          op->setOperands(newOpnds);
        }
      }
    });
  }
};

std::unique_ptr<mlir::Pass> createClosureConversionPass() { return std::make_unique<ClosureConversionPass>(); }
}

#endif // CAKEML_CLOSURE_CONVERSION_H
