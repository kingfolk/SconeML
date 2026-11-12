#ifndef SCONEML_F32_SUPPORT_H
#define SCONEML_F32_SUPPORT_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/OperationSupport.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/Error.h"
#include <stdexcept>
#include <unordered_map>

namespace sconeml::tensor {
inline llvm::APFloat parseF32Literal(llvm::StringRef text) {
  llvm::APFloat value(llvm::APFloat::IEEEsingle());
  auto status =
      value.convertFromString(text, llvm::APFloat::rmNearestTiesToEven);
  if (!status) {
    llvm::consumeError(status.takeError());
    throw std::invalid_argument("invalid f32 literal");
  }
  if (!value.isFinite())
    throw std::invalid_argument("f32 literal outside finite binary32 range");
  if (value.isDenormal())
    value = llvm::APFloat::getZero(value.getSemantics(), value.isNegative());
  return value;
}

// Integer bit tests keep signed FTZ independent of the host FP environment and
// Metal's permitted denormal behavior. Each sequence is verified before
// lowering.
inline mlir::Value flushF32(mlir::OpBuilder &b, mlir::Location loc,
                            mlir::Value value) {
  auto c = [&](int32_t n) {
    return mlir::arith::ConstantIntOp::create(b, loc, n, 32).getResult();
  };
  auto magnitudeMask = c(0x7fffffff);
  auto minNormal = c(0x00800000);
  auto signMask = c(static_cast<int32_t>(0x80000000u));
  auto bits = mlir::arith::BitcastOp::create(b, loc, b.getI32Type(), value);
  auto magnitude = mlir::arith::AndIOp::create(b, loc, bits, magnitudeMask);
  auto tiny = mlir::arith::CmpIOp::create(
      b, loc, mlir::arith::CmpIPredicate::ult, magnitude, minNormal);
  auto sign = mlir::arith::AndIOp::create(b, loc, bits, signMask);
  auto selected = mlir::arith::SelectOp::create(b, loc, tiny, sign, bits);
  return mlir::arith::BitcastOp::create(b, loc, b.getF32Type(), selected);
}

// Metal detects multiplication underflow before rounding. In particular,
// minNormal * nextDown(1) flushes although RNE alone would give minNormal.
// Normal significands are exact 24-bit integers. Their 48-bit product is tiny
// iff E1+E2 < 127, or E1+E2 == 127 and M1*M2 < 2^47.
inline mlir::Value flushF32Product(mlir::OpBuilder &b, mlir::Location loc,
                                   mlir::Value raw, mlir::Value left,
                                   mlir::Value right) {
  auto c32 = [&](uint32_t n) {
    return mlir::arith::ConstantIntOp::create(b, loc, n, 32).getResult();
  };
  auto bits = [&](mlir::Value v) {
    return mlir::arith::BitcastOp::create(b, loc, b.getI32Type(), v)
        .getResult();
  };
  auto exponentMask = c32(0x7f800000), shift = c32(23),
       sigMask = c32(0x007fffff), implicit = c32(0x00800000),
       boundary = c32(127), signMask = c32(0x80000000u);
  auto a = bits(left), d = bits(right);
  auto exponent = [&](mlir::Value v) {
    auto masked = mlir::arith::AndIOp::create(b, loc, v, exponentMask);
    return mlir::arith::ShRUIOp::create(b, loc, masked, shift).getResult();
  };
  auto ea = exponent(a), eb = exponent(d);
  auto sum = mlir::arith::AddIOp::create(b, loc, ea, eb);
  auto below = mlir::arith::CmpIOp::create(
      b, loc, mlir::arith::CmpIPredicate::ult, sum, boundary);
  auto equal = mlir::arith::CmpIOp::create(
      b, loc, mlir::arith::CmpIPredicate::eq, sum, boundary);
  auto significand = [&](mlir::Value v) {
    auto masked = mlir::arith::AndIOp::create(b, loc, v, sigMask);
    auto sig = mlir::arith::OrIOp::create(b, loc, masked, implicit);
    return mlir::arith::ExtUIOp::create(b, loc, b.getI64Type(), sig)
        .getResult();
  };
  auto ma = significand(a), mb = significand(d);
  auto product = mlir::arith::MulIOp::create(b, loc, ma, mb);
  auto limit = mlir::arith::ConstantIntOp::create(b, loc, 1LL << 47, 64);
  auto small = mlir::arith::CmpIOp::create(
      b, loc, mlir::arith::CmpIPredicate::ult, product, limit);
  auto edge = mlir::arith::AndIOp::create(b, loc, equal, small);
  auto tiny = mlir::arith::OrIOp::create(b, loc, below, edge);
  auto resultBits = bits(raw);
  auto sign = mlir::arith::AndIOp::create(b, loc, resultBits, signMask);
  auto selected = mlir::arith::SelectOp::create(b, loc, tiny, sign, resultBits);
  return mlir::arith::BitcastOp::create(b, loc, b.getF32Type(), selected);
}

// Match the complete canonical normalization against a scratch block. No real
// IR is mutated, including use lists. This rejects changed masks, captures,
// missing normalization, flags, and alternate integer/index computations.
inline mlir::Value matchF32Normalization(mlir::Block::iterator &cursor,
                                         mlir::Block::iterator end,
                                         mlir::Value raw, mlir::Value left = {},
                                         mlir::Value right = {}) {
  mlir::Block pattern;
  mlir::OpBuilder b(raw.getContext());
  auto loc = b.getUnknownLoc();
  auto input = pattern.addArgument(b.getF32Type(), loc);
  llvm::DenseMap<mlir::Value, mlir::Value> mapping;
  mapping[input] = raw;
  b.setInsertionPointToEnd(&pattern);
  mlir::Value result;
  if (left) {
    auto a = pattern.addArgument(b.getF32Type(), loc),
         d = pattern.addArgument(b.getF32Type(), loc);
    mapping[a] = left;
    mapping[d] = right;
    result = flushF32Product(b, loc, input, a, d);
  } else
    result = flushF32(b, loc, input);
  for (auto &expected : pattern) {
    if (cursor == end)
      return {};
    auto *actual = &*cursor++;
    if (!mlir::OperationEquivalence::isEquivalentTo(
            &expected, actual,
            [&](mlir::Value a, mlir::Value d) {
              return mlir::success(mapping.lookup(a) == d);
            },
            [&](mlir::Value a, mlir::Value d) { mapping[a] = d; },
            mlir::OperationEquivalence::IgnoreLocations))
      return {};
  }
  return mapping.lookup(result);
}

// Every scalar value is normalized, and no fast-math or foreign operand is
// accepted. The same verifier protects CPU lowering and Metal export.
inline bool validateF32Scalars(mlir::Block::iterator cursor,
                               mlir::Block::iterator end,
                               mlir::ValueRange inputs, mlir::Value output) {
  llvm::DenseSet<mlir::Value> values;
  for (auto input : inputs) {
    auto value = matchF32Normalization(cursor, end, input);
    if (!value)
      return false;
    values.insert(value);
  }
  while (cursor != end) {
    auto *op = &*cursor++;
    if (auto c = mlir::dyn_cast<mlir::arith::ConstantOp>(op)) {
      auto attr = mlir::dyn_cast<mlir::FloatAttr>(c.getValue());
      if (!attr || !c.getType().isF32() || !attr.getValue().isFinite() ||
          attr.getValue().isDenormal())
        return false;
      values.insert(c.getResult());
      continue;
    }
    if (!mlir::isa<mlir::arith::AddFOp, mlir::arith::SubFOp,
                   mlir::arith::MulFOp>(op) ||
        op->getNumOperands() != 2 || op->getNumResults() != 1 ||
        !op->getResult(0).getType().isF32() ||
        !values.contains(op->getOperand(0)) ||
        !values.contains(op->getOperand(1)))
      return false;
    auto flags = mlir::cast<mlir::arith::ArithFastMathInterface>(op);
    if (flags.getFastMathFlagsAttr().getValue() !=
        mlir::arith::FastMathFlags::none)
      return false;
    auto value =
        mlir::isa<mlir::arith::MulFOp>(op)
            ? matchF32Normalization(cursor, end, op->getResult(0),
                                    op->getOperand(0), op->getOperand(1))
            : matchF32Normalization(cursor, end, op->getResult(0));
    if (!value)
      return false;
    values.insert(value);
  }
  return values.contains(output);
}
} // namespace sconeml::tensor
#endif
