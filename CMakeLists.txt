cmake_minimum_required(VERSION 3.20)
project(perf-simulator LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Coverage instrumentation for the CI coverage job (gcov). Off by default so
# local/release builds stay clean.
option(PERF_ENABLE_COVERAGE "Build with gcov coverage instrumentation" OFF)
if(PERF_ENABLE_COVERAGE)
  add_compile_options(--coverage -O0 -g)
  add_link_options(--coverage)
endif()

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
include(CTest)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)

add_definitions(${LLVM_DEFINITIONS})

# googletest for the unit tests. Fetched here rather than taken from LLVM:
# llvm_gtest is an internal build-tree-only target that an installed LLVM
# (used by CI) does not ship. gtest needs RTTI/exceptions, which the rest of
# the project compiles without, so build it in its own scope with them on.
include(FetchContent)
FetchContent_Declare(googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.14.0
  GIT_SHALLOW TRUE)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_subdirectory(include/perf)
add_subdirectory(lib)
add_subdirectory(tools)
add_subdirectory(test)
