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
                 --prior=${PERF_FIXTURE_DIR}/prior.mlir
  )
  set_tests_properties(fixture_${name} PROPERTIES
    WILL_FAIL TRUE)
endfunction()

add_perf_fixture_test(input)

# Same positive fixture through the explicit-prior path.
add_test(
  NAME fixture_input_with_prior
  COMMAND perf --input=${PERF_FIXTURE_DIR}/input.mlir
               --target=${PERF_FIXTURE_DIR}/target.mlir
               --prior=${PERF_FIXTURE_DIR}/prior.mlir
)
set_tests_properties(fixture_input_with_prior PROPERTIES
  PASS_REGULAR_EXPRESSION "with prior: systolic_npu_base.*OK:")
add_perf_fixture_negative_test(r1_unknown_kind)
add_perf_fixture_negative_test(r1_index_range)
add_perf_fixture_negative_test(r1_unknown_space)
add_perf_fixture_negative_test(r1_host_access)

# ---------------------------------------------------------------------------
# --dump-contract: the projected compiler-simulator contract must contain
# every expected element of the design's executor/link vocabulary.
# ---------------------------------------------------------------------------

add_test(
  NAME contract_dump
  COMMAND perf --input=${PERF_FIXTURE_DIR}/input.mlir
               --target=${PERF_FIXTURE_DIR}/target.mlir
               --dump-contract
)
set_tests_properties(contract_dump PROPERTIES
  PASS_REGULAR_EXPRESSION
    "perf.contract_def @systolic_npu_v1.*executor_family<kind = \"dma_read\".*executor_family<kind = \"dma_write\".*executor_family<kind = \"array\".*link<kind = \"dram_bus\"")
