#include "LetAlgDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace sconeml::letalg;

// BORROW FROM lingodb. Not original
ParseResult parseCustRegion(OpAsmParser& parser, Region& result) {
   OpAsmParser::Argument predArgument;
   SmallVector<OpAsmParser::Argument, 4> regionArgs;
   SmallVector<Type, 4> argTypes;
   if (parser.parseLParen()) {
      return failure();
   }
   while (true) {
      Type predArgType;
      if (!parser.parseOptionalRParen()) {
         break;
      }
      if (parser.parseArgument(predArgument) || parser.parseColonType(predArgType)) {
         return failure();
      }
      predArgument.type = predArgType;
      regionArgs.push_back(predArgument);
      if (!parser.parseOptionalComma()) { continue; }
      if (parser.parseRParen()) { return failure(); }
      break;
   }

   if (parser.parseRegion(result, regionArgs)) return failure();
   return success();
}

// BORROW FROM lingodb. Not original
void printCustRegion(OpAsmPrinter& p, Operation* op, Region& r) {
   p << "(";
   bool first = true;
   for (auto arg : r.front().getArguments()) {
      if (first) {
         first = false;
      } else {
         p << ",";
      }
      p << arg << ": " << arg.getType();
   }
   p << ")";
   p.printRegion(r, false, true);
}

//===----------------------------------------------------------------------===//
// LetAlg dialect.
//===----------------------------------------------------------------------===//

// Include the auto-generated definitions.
#include "LetAlgDialectDialect.cpp.inc"

void LetAlgDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "LetAlgDialectOps.cpp.inc"
  >();
}

void LetAlgDialect::registerOps() {
  addOperations<
#define GET_OP_LIST
#include "LetAlgDialectOps.cpp.inc"
  >();
}

Type LetAlgDialect::parseType(DialectAsmParser &parser) const {
    return {};
}

void LetAlgDialect::printType(Type type, DialectAsmPrinter &printer) const {
}

//===----------------------------------------------------------------------===//
// LetAlg Operations
//===----------------------------------------------------------------------===//

// Include the auto-generated operation definitions.
#define GET_OP_CLASSES
#include "LetAlgDialectOps.cpp.inc"

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

void ConstantOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                       double value) {
  auto dataType = builder.getF64Type();
  auto dataAttribute = builder.getF64FloatAttr(value);
  build(builder, state, dataType, dataAttribute);
}

//===----------------------------------------------------------------------===//
// TensorMapOp
//===----------------------------------------------------------------------===//

LogicalResult TensorMapOp::verify() {
  if (getInputs().empty() || getInputs().size() > 3)
    return emitOpError("expects between one and three input buffers");
  auto inputType = dyn_cast<MemRefType>(getInputs().front().getType());
  auto outputType = dyn_cast<MemRefType>(getOutput().getType());
  if (!inputType || !outputType || inputType.getRank() != 1 ||
      outputType.getRank() != 1 || inputType != outputType)
    return emitOpError(
        "expects matching rank-1 input and output memref buffers");
  if (!inputType.getElementType().isSignlessInteger(32))
    return emitOpError("expects i32 input and output elements");
  for (Value input : getInputs()) {
    if (input.getType() != outputType)
      return emitOpError("expects matching rank-1 input and output memref buffers");
    if (input == getOutput())
      return emitOpError("expects distinct input and output buffers");
  }

  if (getBody().empty() || !llvm::hasSingleElement(getBody()))
    return emitOpError("expects a single-block scalar body");
  Block &body = getBody().front();
  if (body.getNumArguments() != getInputs().size() ||
      llvm::any_of(body.getArguments(), [&](BlockArgument argument) {
        return argument.getType() != inputType.getElementType();
      }))
    return emitOpError(
        "expects one body argument matching the memref element type per input");
  auto yield = body.empty() ? YieldOp() : dyn_cast<YieldOp>(&body.back());
  if (!yield || yield.getExpr().getType() != inputType.getElementType())
    return emitOpError("body must yield the memref element type");

  // All dependencies, including the yielded value, must be local to the map.
  // Supporting captures later requires an explicit operand/argument contract.
  for (Operation &operation : body) {
    for (Value operand : operation.getOperands())
      if (operand.getParentBlock() != &body)
        return emitOpError("body must not capture values from outside the map");
    if (&operation == yield.getOperation())
      continue;
    if (!isMemoryEffectFree(&operation))
      return emitOpError("body must be free of memory effects; found ")
             << operation.getName();
    if (!isa<arith::ConstantOp, arith::AddIOp, arith::SubIOp, arith::MulIOp>(
            operation) ||
        operation.getNumResults() != 1 ||
        !operation.getResult(0).getType().isSignlessInteger(32))
      return emitOpError(
                 "body supports only i32 constants and add/sub/mul; found ")
             << operation.getName();
    if (auto flags =
            dyn_cast<arith::ArithIntegerOverflowFlagsInterface>(&operation))
      if (flags.hasNoSignedWrap() || flags.hasNoUnsignedWrap())
        return emitOpError("body arithmetic must use wrapping i32 semantics");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// AddOp
//===----------------------------------------------------------------------===//

// AddOp doesn't need any special implementation beyond what's generated.

//===----------------------------------------------------------------------===//
// PrintOp
//===----------------------------------------------------------------------===//

// PrintOp doesn't need any special implementation beyond what's generated.
