#ifndef EXAMPLE_DIALECT
#define EXAMPLE_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Example dialect definition.
//===----------------------------------------------------------------------===//

def Example_Dialect : Dialect {
  let name = "example";
  let summary = "A simple example dialect";
  let description = [{
    This dialect contains simple operations for demonstration purposes.
  }];
  let cppNamespace = "::mlir::example";
  
  let useDefaultTypePrinterParser = 1;
  let extraClassDeclaration = [{
    void registerOps();
  }];
}

//===----------------------------------------------------------------------===//
// Base example operation definition.
//===----------------------------------------------------------------------===//

class Example_Op<string mnemonic, list<Trait> traits = []> :
        Op<Example_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Example Operations
//===----------------------------------------------------------------------===//

def ConstantOp : Example_Op<"constant", [Pure]> {
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

def AddOp : Example_Op<"add", [Pure]> {
  let summary = "element-wise addition operation";
  let description = [{
    The "add" operation performs element-wise addition between two values
    of the same type.
  }];

  let arguments = (ins F64:$lhs, F64:$rhs);
  let results = (outs F64:$result);

  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

def PrintOp : Example_Op<"print"> {
  let summary = "print operation";
  let description = [{
    The "print" operation prints a given input from within the IR.
  }];

  let arguments = (ins F64:$input);

  let assemblyFormat = "$input attr-dict `:` type($input)";
}

//===----------------------------------------------------------------------===//
// Example Types
// //===----------------------------------------------------------------------===//
// def ExampleType : TypeDef<Example_Dialect, "Example"> {
//     let mnemonic = "example_type";
//     let summary = "An example type";
    
//     // 可以定义参数
//     let parameters = (ins "int":$width);
    
//     // TableGen 会自动生成 parseType 和 printType
// }


#endif // EXAMPLE_DIALECT