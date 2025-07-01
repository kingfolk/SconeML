#ifndef LETALG_DIALECT
#define LETALG_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// LetAlg dialect definition.
//===----------------------------------------------------------------------===//

def LetAlg_Dialect : Dialect {
  let name = "letalg";
  let summary = "A simple let style dialect";
  let description = [{
    This dialect contains simple operations for demonstration purposes.
  }];
  let cppNamespace = "::mlir::letalg";
  
  let useDefaultTypePrinterParser = 1;
  let extraClassDeclaration = [{
    void registerOps();
  }];
}

//===----------------------------------------------------------------------===//
// Base LetAlg operation definition.
//===----------------------------------------------------------------------===//

class LetAlg_Op<string mnemonic, list<Trait> traits = []> :
        Op<LetAlg_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// LetAlg Operations
//===----------------------------------------------------------------------===//

def ConstantOp : LetAlg_Op<"constant", [Pure]> {
  let summary = "constant";
  let description = [{
    Constant operation turns a literal into an SSA value. The data is attached
    to the operation as an attribute. For example:

    ```mlir
      %0 = example.constant 5.5 : f64
    ```
  }];

  let arguments = (ins F64Attr:$value);
  let results = (outs F64:$result);

  let assemblyFormat = "$value attr-dict `:` type($result)";

  let builders = [
    OpBuilder<(ins "double":$value)>
  ];
}

def PrintOp : LetAlg_Op<"print"> {
  let summary = "print operation";
  let description = [{
    The "print" operation prints a given input from within the IR.
  }];

  let arguments = (ins F64:$input);

  let assemblyFormat = "$input attr-dict `:` type($input)";
}

// - let x = ... in let y = ... in ... in x + y + ... could be one LetOp
//   let x = let z ..., `let z ...` should be a standalone letOp
def LetOp : LetAlg_Op<"let", [Pure]> {
  let summary = "let operation";
  let description = [{
    The "let" operation performs a lambda abstraction.
  }];

  let arguments = (ins Variadic<AnyType>: $vars);
  let results = (outs AnyType: $res);
  let regions = (region SizedRegion<1>: $body);

  let assemblyFormat = [{ ($vars^ `:` type($vars))? custom<CustRegion>($body)  `->` type($res) attr-dict-with-keyword }];
}

// - let f x = f x ..., contains one LambdaOp
//   let f x = let f1 y = y + 1 in f1 (x * 2), contains two LambdaOp
//     first: LambdaOp(arguments: <y>, regions: {y+1})
//     second: LambdaOp(arguments: <x>, regions: {ApplyOp(f1, x*2)})
//   let f x = let f1 y = x * 2 + 1 in f1, contains two LambdaOp
//     first: LambdaOp(arguments: <y, x>, regions: {x*2+1})
//     second: LambdaOp(arguments: <x>, regions: <f1>). second LetOp return a LetOp
def LambdaOp : LetAlg_Op<"lambda", [Pure]> {
  let summary = "lambda operation";
  let description = [{
    The "lambda" operation defines a lambda abstraction.
  }];

  let arguments = (ins StrAttr: $name, Variadic<AnyType>: $vars);
  let results = (outs AnyType: $res);
  let regions = (region SizedRegion<1>: $body);

  let assemblyFormat = [{ $name ($vars^ `:` type($vars))? custom<CustRegion>($body)  `->` type($res) attr-dict-with-keyword }];
}

def YieldOp : LetAlg_Op<"yield", [Pure,Terminator]> {
  let summary = "yield operation";
  let description = [{
    The "yield" operations yield/return a value.
  }];

  let arguments = (ins AnyType: $expr);
  let results = (outs AnyType: $res);
}

def ApplyOp : LetAlg_Op<"apply", [Pure]> {
  let summary = "apply operation";
  let description = [{
    The "apply" operations perform apply arguments to a let op.
  }];

  let arguments = (ins AnyType: $fn, Variadic<AnyType>: $vars);
  let results = (outs AnyType: $res);
}

//===----------------------------------------------------------------------===//
// LetAlg Types
// //===----------------------------------------------------------------------===//
// def LetAlgType : TypeDef<LetAlg_Dialect, "LetAlg"> {
//     let mnemonic = "letalg_type";
//     let summary = "An letalg type";
    
//     let parameters = (ins "int":$width);
    
//     // TableGen auto gen parseType, printType
// }


#endif // LETALG_DIALECT