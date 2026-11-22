# ---------------------------------------------------------------------------
# Regression (acceptance) tests: end-to-end runs of the perf tool against
# fixture files. These cover everything that depends on the MLIR front-end
# (parsing, op verifiers, R1..Rn rules, contract projection).
#
# Layout:
#   config/                     shared design + prior configs
#     target.mlir               the canonical design point
#     prior.mlir                the canonical prior (calibration state)
#   <group>/                    one subdirectory per feature area
#     <name>.mlir               must parse+verify (exit 0, "OK:")
#     <name>.invalid.mlir       must be rejected (non-zero exit)
# ---------------------------------------------------------------------------

set(PERF_FIXTURE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(PERF_CONFIG_DIR  ${PERF_FIXTURE_DIR}/config)

function(add_perf_fixture_test group name)
  add_test(
    NAME fixture_${group}_${name}
    COMMAND perf --input=${PERF_FIXTURE_DIR}/${group}/${name}.mlir
                 --target=${PERF_CONFIG_DIR}/target.mlir
  )
  set_tests_properties(fixture_${group}_${name} PROPERTIES
    PASS_REGULAR_EXPRESSION "OK:")
endfunction()

function(add_perf_fixture_negative_test group name)
  add_test(
    NAME fixture_${group}_${name}
    COMMAND perf --input=${PERF_FIXTURE_DIR}/${group}/${name}.invalid.mlir
                 --target=${PERF_CONFIG_DIR}/target.mlir
  )
  set_tests_properties(fixture_${group}_${name} PROPERTIES
    WILL_FAIL TRUE)
endfunction()

# --- Reachability (R1) -----------------------------------------------------

add_perf_fixture_test(reachability input)
add_perf_fixture_test(reachability valid)
add_perf_fixture_negative_test(reachability violations)
set_tests_properties(fixture_reachability_violations PROPERTIES
  FAIL_REGULAR_EXPRESSION "R1: unknown executor kind 'dma_rade';R1: executor 'dma_read#7' out of range;R1: unknown memory space 'l2';R1: unknown executor kind 'dram_bus'")

# Same positive fixture through the explicit-prior path.
add_test(
  NAME fixture_reachability_input_with_prior
  COMMAND perf --input=${PERF_FIXTURE_DIR}/reachability/input.mlir
               --target=${PERF_CONFIG_DIR}/target.mlir
               --prior=${PERF_CONFIG_DIR}/prior.mlir
)
set_tests_properties(fixture_reachability_input_with_prior PROPERTIES
  PASS_REGULAR_EXPRESSION "with prior: systolic_npu_base.*OK:")

# --- Contract projection ---------------------------------------------------

add_test(
  NAME contract_dump
  COMMAND perf --input=${PERF_FIXTURE_DIR}/reachability/input.mlir
               --target=${PERF_CONFIG_DIR}/target.mlir
               --dump-contract
)
set_tests_properties(contract_dump PROPERTIES
  PASS_REGULAR_EXPRESSION
    "perf.contract_def @systolic_npu_v1.*executor_family<kind = \"dma_read\".*executor_family<kind = \"dma_write\".*executor_family<kind = \"array\".*link<kind = \"dram_bus\"")
