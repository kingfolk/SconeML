cmake_minimum_required(VERSION 3.20)
project(MLIRExampleProject)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找LLVM和MLIR
# find_package(LLVM REQUIRED CONFIG)
# find_package(MLIR REQUIRED CONFIG)

# message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
# message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")

find_package(LLVM 20.1 REQUIRED)
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
set(MLIR_DIR "${LLVM_DIR}/../mlir")
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
find_package(MLIR 20.1 REQUIRED)

# 设置LLVM定义和包含目录
separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
add_definitions(${LLVM_DEFINITIONS_LIST})

# 添加LLVM和MLIR到CMake路径
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")

# 包含必要的CMake模块
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

# 设置包含目录
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# 设置编译定义
add_definitions(${LLVM_DEFINITIONS})

# TableGen规则
set(LLVM_TARGET_DEFINITIONS ExampleDialect.td)

mlir_tablegen(ExampleDialectDialect.h.inc -gen-dialect-decls)
mlir_tablegen(ExampleDialectDialect.cpp.inc -gen-dialect-defs)
mlir_tablegen(ExampleDialectOps.h.inc -gen-op-decls)
mlir_tablegen(ExampleDialectOps.cpp.inc -gen-op-defs)
# mlir_tablegen(ExampleTypes.h.inc -gen-typedef-decls)
# mlir_tablegen(ExampleTypes.cpp.inc -gen-typedef-defs)

add_public_tablegen_target(ExampleDialectIncGen)

# 获取LLVM组件
# llvm_map_components_to_libnames(llvm_libs support core irreader)

# 创建可执行文件
add_llvm_executable(mlir_example
  main.cpp
  ExampleDialect.cpp
)

# 添加依赖关系
add_dependencies(mlir_example ExampleDialectIncGen)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

# 链接库
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

# 设置包含目录
target_include_directories(mlir_example 
  PRIVATE 
  ${CMAKE_CURRENT_BINARY_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${MLIR_INCLUDE_DIRS}
  ${LLVM_INCLUDE_DIRS}
)