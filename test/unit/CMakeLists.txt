add_custom_target(PerfUnitTests)
set_target_properties(PerfUnitTests PROPERTIES FOLDER "Perf tests")

# Unit tests: standalone C++ components that do not exercise the MLIR
# front-end (config structures, derived quantities, pure analyses).
function(add_perf_unittest test_name)
  add_unittest(PerfUnitTests ${test_name} ${ARGN})
  # Register with ctest so a single `ctest` run covers UTs and fixtures.
  add_test(NAME ${test_name} COMMAND ${test_name})
endfunction()

add_subdirectory(analysis)
add_subdirectory(sim)
