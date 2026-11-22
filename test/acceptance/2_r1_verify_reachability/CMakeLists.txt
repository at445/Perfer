# ---------------------------------------------------------------------------
# Reachability (R1) acceptance: every op's memref operands must reside in a
# memory space reachable by its placement resource, checked against the
# projected contract.
# ---------------------------------------------------------------------------

add_perf_fixture_test(2_r1_verify_reachability valid)
add_perf_fixture_negative_test(2_r1_verify_reachability violations)
set_tests_properties(fixture_r1_verify_reachability_violations PROPERTIES
  FAIL_REGULAR_EXPRESSION "R1: unknown executor kind 'dma_rade';R1: executor 'dma_read#7' out of range;R1: unknown memory space 'l2';R1: unknown executor kind 'dram_bus'")
