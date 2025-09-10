#ifndef SCONEML_LETALG_DIALECT_H
#define SCONEML_LETALG_DIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

// Include the auto-generated header file containing the declaration of our dialect.
#include "LetAlgDialectDialect.h.inc"

// Include the auto-generated header file containing the declarations of our operations.
#define GET_OP_CLASSES
#include "LetAlgDialectOps.h.inc"

#endif // SCONEML_LETALG_DIALECT_H