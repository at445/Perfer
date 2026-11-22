# ---------------------------------------------------------------------------
# Schedule legality (R3) acceptance: the per-resource total order declared
# by perf.schedule entries must be a linear extension of the token partial
# order; violations replay-deadlock.
# ---------------------------------------------------------------------------

add_perf_fixture_test(4_r3_verify_schedule valid)
add_perf_fixture_negative_test(4_r3_verify_schedule violations)
set_tests_properties(fixture_r3_verify_schedule_violations PROPERTIES
  FAIL_REGULAR_EXPRESSION "R3: schedule order on resource 'dma_read#0' conflicts with token dependency")
