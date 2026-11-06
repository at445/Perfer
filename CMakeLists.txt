cmake_minimum_required(VERSION 3.20)
project(perf-simulator LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------------------------------------------------------------------------
# LLVM/MLIR discovery
# Priority:
#   1. LLVM_BUILD_DIR env var (developer override)
#   2. Common sibling paths (../llvm-project/build, ../../llvm-project/build)
#   3. CMAKE_PREFIX_PATH / system paths
# ---------------------------------------------------------------------------

if(DEFINED ENV{LLVM_BUILD_DIR})
  set(_llvm_build "$ENV{LLVM_BUILD_DIR}")
else()
  # Try common relative locations.
  get_filename_component(_src_dir "${CMAKE_SOURCE_DIR}" REALPATH)
  foreach(_candidate
      "${_src_dir}/../llvm-project/build"
      "${_src_dir}/../../llvm-project/build"
      "${_src_dir}/../llvm/build"
      "${_src_dir}/../../llvm/build")
    if(EXISTS "${_candidate}/lib/cmake/mlir/MLIRConfig.cmake")
      set(_llvm_build "${_candidate}")
      break()
    endif()
  endforeach()
endif()

if(_llvm_build)
  set(MLIR_DIR "${_llvm_build}/lib/cmake/mlir" CACHE PATH "MLIR cmake dir")
  set(LLVM_DIR "${_llvm_build}/lib/cmake/llvm" CACHE PATH "LLVM cmake dir")
  set(CMAKE_C_COMPILER "${_llvm_build}/bin/clang" CACHE FILEPATH "C compiler")
  set(CMAKE_CXX_COMPILER "${_llvm_build}/bin/clang++" CACHE FILEPATH "CXX compiler")
  message(STATUS "Found LLVM build at: ${_llvm_build}")
endif()

find_package(MLIR REQUIRED CONFIG)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)

add_definitions(${LLVM_DEFINITIONS})

add_subdirectory(include/perf)
add_subdirectory(lib)
add_subdirectory(tools)
add_subdirectory(test)
