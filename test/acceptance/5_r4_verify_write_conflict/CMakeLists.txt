# ---------------------------------------------------------------------------
# Write conflict (R4) acceptance: conflicting accesses (same root buffer,
# overlapping ranges) with at least one write must be ordered by the token
# partial order; unordered write conflicts replay to undefined behavior.
# ---------------------------------------------------------------------------

add_perf_fixture_test(5_r4_verify_write_conflict valid)
add_perf_fixture_negative_test(5_r4_verify_write_conflict violations)
set_tests_properties(fixture_r4_verify_write_conflict_violations PROPERTIES
  FAIL_REGULAR_EXPRESSION "R4: write conflict between unordered accesses")
