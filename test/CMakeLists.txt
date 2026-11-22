add_custom_target(PerfUnitTests)
set_target_properties(PerfUnitTests PROPERTIES FOLDER "Perf tests")

function(add_perf_unittest test_name)
  add_unittest(PerfUnitTests ${test_name} ${ARGN})
  # Register with ctest so a single `ctest` run covers UTs and fixtures.
  add_test(NAME ${test_name} COMMAND ${test_name})
endfunction()

add_subdirectory(Analysis)
add_subdirectory(Sim)

# ---------------------------------------------------------------------------
# Fixture acceptance tests (lit-style, driven by ctest)
#
# Convention under test/fixtures/:
#   - <name>.mlir            a program that must parse+verify against
#                            test/fixtures/target.mlir (exit code 0)
#   - <name>.invalid.mlir    a program that must be rejected (non-zero exit)
# ---------------------------------------------------------------------------

set(PERF_FIXTURE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/fixtures)

function(add_perf_fixture_test name)
  add_test(
    NAME fixture_${name}
    COMMAND perf --input=${PERF_FIXTURE_DIR}/${name}.mlir
                 --target=${PERF_FIXTURE_DIR}/target.mlir
  )
  set_tests_properties(fixture_${name} PROPERTIES
    PASS_REGULAR_EXPRESSION "OK:")
endfunction()

function(add_perf_fixture_negative_test name)
  add_test(
    NAME fixture_${name}
    COMMAND perf --input=${PERF_FIXTURE_DIR}/${name}.invalid.mlir
                 --target=${PERF_FIXTURE_DIR}/target.mlir
  )
  set_tests_properties(fixture_${name} PROPERTIES
    WILL_FAIL TRUE)
endfunction()

add_perf_fixture_test(input)
