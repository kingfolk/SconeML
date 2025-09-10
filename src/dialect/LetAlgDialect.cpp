#include "LetAlgDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"
#include "llvm/ADT/TypeSwitch.h"

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
// AddOp
//===----------------------------------------------------------------------===//

// AddOp doesn't need any special implementation beyond what's generated.

//===----------------------------------------------------------------------===//
// PrintOp
//===----------------------------------------------------------------------===//

// PrintOp doesn't need any special implementation beyond what's generated.
