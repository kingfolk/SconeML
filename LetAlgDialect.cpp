#include "LetAlgDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::letalg;

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