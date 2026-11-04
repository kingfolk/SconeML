cmake_minimum_required(VERSION 3.20)
project(MLIRSconemlProject)

if(APPLE)
  enable_language(OBJCXX)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(LLVM 22.1 REQUIRED)
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
set(MLIR_DIR "${LLVM_DIR}/../mlir")
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
find_package(MLIR 22.1 REQUIRED)

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
set(LLVM_TARGET_DEFINITIONS src/dialect/LetAlgDialect.td)

mlir_tablegen(LetAlgDialectDialect.h.inc -gen-dialect-decls)
mlir_tablegen(LetAlgDialectDialect.cpp.inc -gen-dialect-defs)
mlir_tablegen(LetAlgDialectOps.h.inc -gen-op-decls)
mlir_tablegen(LetAlgDialectOps.cpp.inc -gen-op-defs)
# mlir_tablegen(LetAlgTypes.h.inc -gen-typedef-decls)
# mlir_tablegen(LetAlgTypes.cpp.inc -gen-typedef-defs)

add_public_tablegen_target(LetAlgDialectIncGen)

add_llvm_executable(mlir_example
  src/main.cpp
  src/dialect/LetAlgDialect.cpp
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
  src/dialect/LetAlgDialect.cpp
)

target_compile_options(tester PRIVATE -fexceptions)

add_dependencies(tester LetAlgDialectIncGen)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

target_link_libraries(tester
  PRIVATE
  MLIR
  MLIRExecutionEngine
)

target_include_directories(tester 
  PRIVATE 
  ${CMAKE_CURRENT_BINARY_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${MLIR_INCLUDE_DIRS}
  ${LLVM_INCLUDE_DIRS}
)

if(APPLE)
  set(SCONEML_GPU_RUNTIME src/tensor/MetalRuntime.mm)
  set_source_files_properties(src/tensor/MetalRuntime.mm
    PROPERTIES COMPILE_FLAGS "-fobjc-arc")
else()
  set(SCONEML_GPU_RUNTIME src/tensor/MetalRuntimeStub.cpp)
endif()
# Pipeline libraries are also consumed by the unified test runner.
add_library(SconeMLTensorGPU STATIC src/conversion/LowerTensorToGPU.cpp)
add_dependencies(SconeMLTensorGPU LetAlgDialectIncGen)
target_link_libraries(SconeMLTensorGPU PUBLIC
  MLIR)

add_library(SconeMLMetalExporter STATIC src/conversion/MetalExporter.cpp)
target_link_libraries(SconeMLMetalExporter PUBLIC
  MLIR)

add_library(SconeMLTestSpec STATIC test/runner/TestSpec.cpp)
target_compile_options(SconeMLTestSpec PRIVATE -fexceptions)
target_link_libraries(SconeMLTestSpec PUBLIC LLVM)

if(APPLE)
  set(SCONEML_TEST_METAL test/runner/MetalAdapter.mm)
  set_source_files_properties(test/runner/MetalAdapter.mm PROPERTIES COMPILE_FLAGS "-fobjc-arc")
else()
  set(SCONEML_TEST_METAL test/runner/MetalAdapterStub.cpp)
endif()
target_sources(tester PRIVATE
  test/runner/Pipelines.cpp
  test/runner/Assertions.cpp
  src/tensor/TensorFrontend.cpp
  src/tensor/CPURunner.cpp
  src/tensor/GPURunner.cpp
  src/tensor/MetalLaunch.cpp
  ${SCONEML_GPU_RUNTIME}
  ${SCONEML_TEST_METAL})
target_compile_definitions(tester PRIVATE SCONEML_TEST_ROOT="${CMAKE_CURRENT_SOURCE_DIR}/test")
target_link_libraries(tester PRIVATE
  SconeMLTestSpec SconeMLTensorGPU SconeMLMetalExporter $<TARGET_FILE:LLVMFileCheck>)
# ExecutionEngine uses the monolithic MLIR/LLVM libraries in this distribution.
# Avoid mixing component archives (and duplicate interface TypeIDs) with MLIR.
# FileCheck itself is a separate archive; its Support symbols come from LLVM.
add_dependencies(tester LLVMFileCheck)
if(APPLE)
  target_link_libraries(tester PRIVATE "-framework Foundation" "-framework Metal")
endif()

# Keep only interface-level contracts that cannot be expressed as program data.
# This target intentionally does not link CPU runner/ExecutionEngine.
add_llvm_executable(tensor_interfaces
  test/runner/Interfaces.cpp
  src/tensor/TensorFrontend.cpp
  src/dialect/LetAlgDialect.cpp)
target_compile_options(tensor_interfaces PRIVATE -fexceptions)
add_dependencies(tensor_interfaces LetAlgDialectIncGen)
target_link_libraries(tensor_interfaces PRIVATE
  SconeMLTestSpec SconeMLTensorGPU SconeMLMetalExporter MLIR)

add_executable(metal_runtime_checks
  test/runner/MetalRuntimeChecks.cpp
  src/tensor/MetalLaunch.cpp
  ${SCONEML_GPU_RUNTIME})
target_compile_options(metal_runtime_checks PRIVATE -fexceptions)
if(APPLE)
  target_link_libraries(metal_runtime_checks PRIVATE "-framework Foundation" "-framework Metal")
endif()

enable_testing()
add_test(NAME compiler_suite COMMAND tester)
add_test(NAME tensor_interfaces COMMAND tensor_interfaces
  ${CMAKE_CURRENT_SOURCE_DIR}/test/tensor/polynomial.ml
  ${CMAKE_CURRENT_SOURCE_DIR}/test/metal_exporter/shared_square.mlir)
add_test(NAME metal_runtime_contracts COMMAND metal_runtime_checks)
add_test(NAME runner_contracts COMMAND ${CMAKE_COMMAND}
  -DRUNNER=$<TARGET_FILE:tester>
  -DDATA_ROOT=${CMAKE_CURRENT_SOURCE_DIR}/test
  -DSCRATCH=${CMAKE_CURRENT_BINARY_DIR}/runner-contracts
  -P ${CMAKE_CURRENT_SOURCE_DIR}/test/runner/RunnerChecks.cmake)
