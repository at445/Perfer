add_custom_target(PerfUnitTests)
set_target_properties(PerfUnitTests PROPERTIES FOLDER "Perf tests")

# googletest is fetched by the project itself (top-level CMake), NOT taken
# from LLVM's build tree: `llvm_gtest` is an internal build-tree-only target
# and is absent from an installed LLVM, which is what CI consumes.

# Unit tests: standalone C++ components that do not exercise the MLIR
# front-end (config structures, derived quantities, pure analyses).
function(add_perf_unittest test_name)
  add_executable(${test_name} ${ARGN})
  add_dependencies(PerfUnitTests ${test_name})
  target_link_libraries(${test_name} PRIVATE gtest_main LLVMSupport)
  # Build UTs as part of the default target so a plain `cmake --build`
  # produces them (ctest then finds the executables).
  set_target_properties(${test_name} PROPERTIES EXCLUDE_FROM_ALL OFF)
  # Register with ctest so a single `ctest` run covers UTs and fixtures.
  add_test(NAME ${test_name} COMMAND ${test_name})
endfunction()

add_subdirectory(analysis)
add_subdirectory(sim)
