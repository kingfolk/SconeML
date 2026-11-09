cmake_minimum_required(VERSION 3.20)
project(MLIRSconemlProject)
find_package(Python3 COMPONENTS Interpreter REQUIRED)

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
add_library(SconeMLTensorGPU STATIC src/conversion/LowerTensorToGPU.cpp
  src/conversion/LowerShapedTensorToGPU.cpp)
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
  test/runner/MapExecution.cpp
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

add_executable(metal_stub_checks
  test/runner/MetalRuntimeChecks.cpp
  src/tensor/MetalLaunch.cpp
  src/tensor/MetalRuntimeStub.cpp)
target_compile_options(metal_stub_checks PRIVATE -fexceptions)
target_compile_definitions(metal_stub_checks PRIVATE SCONEML_EXPECT_METAL_STUB)

add_llvm_executable(cpu_input_contracts
  test/runner/CPUInputContracts.cpp
  src/tensor/CPURunner.cpp
  src/tensor/TensorFrontend.cpp
  src/dialect/LetAlgDialect.cpp)
target_compile_options(cpu_input_contracts PRIVATE -fexceptions)
add_dependencies(cpu_input_contracts LetAlgDialectIncGen)
target_link_libraries(cpu_input_contracts PRIVATE SconeMLTestSpec MLIR MLIRExecutionEngine)

function(add_gpu_input_contract_target target runtime)
  add_llvm_executable(${target}
    test/runner/GPUInputContracts.cpp
    src/tensor/GPURunner.cpp
    src/tensor/TensorFrontend.cpp
    src/tensor/MetalLaunch.cpp
    src/dialect/LetAlgDialect.cpp
    ${runtime})
  target_compile_options(${target} PRIVATE -fexceptions)
  add_dependencies(${target} LetAlgDialectIncGen)
  target_link_libraries(${target} PRIVATE
    SconeMLTestSpec SconeMLTensorGPU SconeMLMetalExporter MLIR)
endfunction()
add_gpu_input_contract_target(gpu_input_contracts ${SCONEML_GPU_RUNTIME})
if(APPLE)
  target_link_libraries(gpu_input_contracts PRIVATE "-framework Foundation" "-framework Metal")
else()
  target_compile_definitions(gpu_input_contracts PRIVATE SCONEML_EXPECT_METAL_STUB)
endif()
add_gpu_input_contract_target(gpu_input_stub_contracts src/tensor/MetalRuntimeStub.cpp)
target_compile_definitions(gpu_input_stub_contracts PRIVATE SCONEML_EXPECT_METAL_STUB)

add_executable(grouped_spec_checks test/runner/GroupedSpecChecks.cpp)
target_compile_options(grouped_spec_checks PRIVATE -fexceptions)
target_link_libraries(grouped_spec_checks PRIVATE SconeMLTestSpec)

add_llvm_executable(shaped_cpu_contracts
  test/runner/ShapedCPUContracts.cpp
  src/tensor/CPURunner.cpp
  src/tensor/TensorFrontend.cpp
  src/dialect/LetAlgDialect.cpp)
target_compile_options(shaped_cpu_contracts PRIVATE -fexceptions)
add_dependencies(shaped_cpu_contracts LetAlgDialectIncGen)
target_link_libraries(shaped_cpu_contracts PRIVATE SconeMLTestSpec MLIR MLIRExecutionEngine)

function(add_shaped_gpu_contract_target target runtime)
  add_llvm_executable(${target}
    test/runner/ShapedGPUContracts.cpp
    test/runner/BroadcastGPUContracts.cpp
    src/tensor/GPURunner.cpp
    src/tensor/TensorFrontend.cpp
    src/tensor/MetalLaunch.cpp
    src/dialect/LetAlgDialect.cpp
    ${runtime})
  target_compile_options(${target} PRIVATE -fexceptions)
  add_dependencies(${target} LetAlgDialectIncGen)
  target_link_libraries(${target} PRIVATE SconeMLTestSpec SconeMLTensorGPU SconeMLMetalExporter MLIR)
endfunction()
add_shaped_gpu_contract_target(shaped_gpu_contracts ${SCONEML_GPU_RUNTIME})
if(APPLE)
  target_link_libraries(shaped_gpu_contracts PRIVATE "-framework Foundation" "-framework Metal")
else()
  target_compile_definitions(shaped_gpu_contracts PRIVATE SCONEML_EXPECT_METAL_STUB)
endif()
add_shaped_gpu_contract_target(shaped_gpu_stub_contracts src/tensor/MetalRuntimeStub.cpp)
target_compile_definitions(shaped_gpu_stub_contracts PRIVATE SCONEML_EXPECT_METAL_STUB)

enable_testing()
add_test(NAME shaped_gpu_contracts COMMAND shaped_gpu_contracts
  ${CMAKE_CURRENT_SOURCE_DIR}/test/shaped_gpu/semantics.ml
  ${CMAKE_CURRENT_SOURCE_DIR}/test/broadcast_gpu/semantics.ml)
add_test(NAME shaped_gpu_stub_contracts COMMAND shaped_gpu_stub_contracts
  ${CMAKE_CURRENT_SOURCE_DIR}/test/shaped_gpu/semantics.ml
  ${CMAKE_CURRENT_SOURCE_DIR}/test/broadcast_gpu/semantics.ml)
add_test(NAME shaped_gpu_reference_fixtures COMMAND ${Python3_EXECUTABLE}
  ${CMAKE_CURRENT_SOURCE_DIR}/test/reference/generate_shaped_gpu.py --check)
add_test(NAME shaped_cpu_contracts COMMAND shaped_cpu_contracts
  ${CMAKE_CURRENT_SOURCE_DIR}/test/shaped_input/same_shape.ml
  ${CMAKE_CURRENT_SOURCE_DIR}/test/shaped_input/broadcast/semantics.ml
  ${CMAKE_CURRENT_SOURCE_DIR}/test/shaped_input/broadcast/empty.ml)
add_test(NAME grouped_spec_contracts COMMAND grouped_spec_checks
  ${CMAKE_CURRENT_SOURCE_DIR}/test/acceptance_m2/matrix/single-input-launch-boundaries.ml)
add_test(NAME compiler_suite COMMAND tester)
add_test(NAME tensor_interfaces COMMAND tensor_interfaces
  ${CMAKE_CURRENT_SOURCE_DIR}/test/tensor/polynomial.ml
  ${CMAKE_CURRENT_SOURCE_DIR}/test/metal_exporter/shared_square.mlir)
add_test(NAME metal_runtime_contracts COMMAND metal_runtime_checks)
add_test(NAME metal_stub_contracts COMMAND metal_stub_checks)
add_test(NAME cpu_input_contracts COMMAND cpu_input_contracts
  ${CMAKE_CURRENT_SOURCE_DIR}/test/external_input/polynomial.ml
  ${CMAKE_CURRENT_SOURCE_DIR}/test/multi_input/alias.ml
  ${CMAKE_CURRENT_SOURCE_DIR}/test/acceptance_m2/lifetime.ml
  ${CMAKE_CURRENT_SOURCE_DIR}/test/acceptance_m2/overlap.ml)
add_test(NAME gpu_input_contracts COMMAND gpu_input_contracts
  ${CMAKE_CURRENT_SOURCE_DIR}/test/external_input/polynomial.ml
  ${CMAKE_CURRENT_SOURCE_DIR}/test/multi_input/alias.ml
  ${CMAKE_CURRENT_SOURCE_DIR}/test/acceptance_m2/lifetime.ml
  ${CMAKE_CURRENT_SOURCE_DIR}/test/acceptance_m2/overlap.ml)
add_test(NAME gpu_input_stub_contracts COMMAND gpu_input_stub_contracts
  ${CMAKE_CURRENT_SOURCE_DIR}/test/external_input/polynomial.ml
  ${CMAKE_CURRENT_SOURCE_DIR}/test/multi_input/alias.ml
  ${CMAKE_CURRENT_SOURCE_DIR}/test/acceptance_m2/lifetime.ml
  ${CMAKE_CURRENT_SOURCE_DIR}/test/acceptance_m2/overlap.ml)
add_test(NAME reference_fixtures COMMAND ${Python3_EXECUTABLE}
  ${CMAKE_CURRENT_SOURCE_DIR}/test/reference/generate.py --check)
add_test(NAME multi_input_reference_fixtures COMMAND ${Python3_EXECUTABLE}
  ${CMAKE_CURRENT_SOURCE_DIR}/test/reference/generate_m2.py --check)
add_test(NAME broadcast_reference_fixtures COMMAND ${Python3_EXECUTABLE}
  ${CMAKE_CURRENT_SOURCE_DIR}/test/reference/generate_broadcast.py --check)
add_test(NAME gpu_broadcast_reference_fixtures COMMAND ${Python3_EXECUTABLE}
  ${CMAKE_CURRENT_SOURCE_DIR}/test/reference/generate_gpu_broadcast.py --check)
add_test(NAME shaped_acceptance_reference_fixtures COMMAND ${Python3_EXECUTABLE}
  ${CMAKE_CURRENT_SOURCE_DIR}/test/reference/generate_m3.py --check)
add_test(NAME runner_contracts COMMAND ${CMAKE_COMMAND}
  -DRUNNER=$<TARGET_FILE:tester>
  -DDATA_ROOT=${CMAKE_CURRENT_SOURCE_DIR}/test
  -DSCRATCH=${CMAKE_CURRENT_BINARY_DIR}/runner-contracts
  -P ${CMAKE_CURRENT_SOURCE_DIR}/test/runner/RunnerChecks.cmake)
