# ---------------------------------------------------------------------------
# Base acceptance: canonical end-to-end run of the perf pipeline. The
# canonical design point + canonical program must parse, verify, and produce
# the verified module. This is the smoke test every other group builds on.
# ---------------------------------------------------------------------------

add_perf_fixture_test(0_base input)

# Same positive fixture through the explicit-prior path.
add_test(
  NAME fixture_base_input_with_prior
  COMMAND perf --input=${CMAKE_CURRENT_SOURCE_DIR}/input.mlir
               --target=${CMAKE_CURRENT_SOURCE_DIR}/target.mlir
               --prior=${CMAKE_CURRENT_SOURCE_DIR}/prior.mlir
)
set_tests_properties(fixture_base_input_with_prior PROPERTIES
  PASS_REGULAR_EXPRESSION "with prior: systolic_npu_base.*OK:")
