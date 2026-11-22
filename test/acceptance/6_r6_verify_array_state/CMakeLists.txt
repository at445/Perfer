# ---------------------------------------------------------------------------
# Array PE-state provenance (R6) acceptance: every compute's PE state must
# trace back to array_init or an earlier compute along the token chain, and
# array_init (operand form) must be ordered after the writeback it reads.
# ---------------------------------------------------------------------------

add_perf_fixture_test(6_r6_verify_array_state valid)
add_perf_fixture_negative_test(6_r6_verify_array_state violations)
set_tests_properties(fixture_r6_verify_array_state_violations PROPERTIES
  FAIL_REGULAR_EXPRESSION "R6: compute accumulates onto PE state of unknown origin;R6: array_init reads old partial sums.*no compute on that array is ordered before it")
