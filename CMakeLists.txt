cmake_minimum_required(VERSION 3.20)
project(MLIRSconemlProject)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(LLVM 20.1 REQUIRED)
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
set(MLIR_DIR "${LLVM_DIR}/../mlir")
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
find_package(MLIR 20.1 REQUIRED)

# Add cmake path
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")

# include lib
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

# include dir
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_definitions(${LLVM_DEFINITIONS})

# tablegen definition
set(LLVM_TARGET_DEFINITIONS LetAlgDialect.td)

mlir_tablegen(LetAlgDialectDialect.h.inc -gen-dialect-decls)
mlir_tablegen(LetAlgDialectDialect.cpp.inc -gen-dialect-defs)
mlir_tablegen(LetAlgDialectOps.h.inc -gen-op-decls)
mlir_tablegen(LetAlgDialectOps.cpp.inc -gen-op-defs)
# mlir_tablegen(LetAlgTypes.h.inc -gen-typedef-decls)
# mlir_tablegen(LetAlgTypes.cpp.inc -gen-typedef-defs)

add_public_tablegen_target(LetAlgDialectIncGen)

add_llvm_executable(mlir_example
  main.cpp
  LetAlgDialect.cpp
)

target_compile_options(mlir_example PRIVATE -fexceptions)

add_dependencies(mlir_example LetAlgDialectIncGen)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

target_link_libraries(mlir_example
  PRIVATE
  ${dialect_libs}
  ${extension_libs}
  ${conversion_libs}
  ${translation_libs}
  MLIRDialect
  MLIRIR
  MLIRParser
  MLIRSupport
  MLIRTransforms
  MLIRTransformUtils
  MLIRAnalysis
  MLIRExecutionEngine
)

target_include_directories(mlir_example 
  PRIVATE 
  ${CMAKE_CURRENT_BINARY_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${MLIR_INCLUDE_DIRS}
  ${LLVM_INCLUDE_DIRS}
)

add_llvm_executable(tester
  test/tester.cpp
  LetAlgDialect.cpp
)

target_compile_options(tester PRIVATE -fexceptions)

add_dependencies(tester LetAlgDialectIncGen)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

target_link_libraries(tester
  PRIVATE
  ${dialect_libs}
  ${extension_libs}
  ${conversion_libs}
  ${translation_libs}
  MLIRDialect
  MLIRIR
  MLIRParser
  MLIRSupport
  MLIRTransforms
  MLIRTransformUtils
  MLIRAnalysis
  MLIRExecutionEngine
)

target_include_directories(tester 
  PRIVATE 
  ${CMAKE_CURRENT_BINARY_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${MLIR_INCLUDE_DIRS}
  ${LLVM_INCLUDE_DIRS}
)