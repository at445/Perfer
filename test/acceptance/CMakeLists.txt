# ---------------------------------------------------------------------------
# Regression (acceptance) tests: end-to-end runs of the perf tool against
# fixture files. These cover everything that depends on the MLIR front-end
# (parsing, op verifiers, R1..Rn rules, contract projection).
#
# Each group directory is self-contained: it carries the target/prior
# configs it runs against plus its program fixtures. Numeric prefixes give
# the groups a stable reading order.
#
#   <N>_<group>/
#     target.mlir               design point this group runs against
#     prior.mlir                prior config (when the group exercises it)
#     CMakeLists.txt            the group's test registrations
#     <name>.mlir               must parse+verify (exit 0, "OK:")
#     <name>.invalid.mlir       must be rejected (non-zero exit)
#
# Groups:
#   0_base/                  canonical end-to-end run (smoke test)
#   1_contract/              --dump-contract projection (expected.mlir golden)
#   3_verify_reachability/   R1: operands reachable from placement resources
# ---------------------------------------------------------------------------

function(add_perf_fixture_test group name)
  string(REGEX REPLACE "^[0-9]+_" "" short_group ${group})
  add_test(
    NAME fixture_${short_group}_${name}
    COMMAND perf --input=${CMAKE_CURRENT_SOURCE_DIR}/${name}.mlir
                 --target=${CMAKE_CURRENT_SOURCE_DIR}/target.mlir
  )
  set_tests_properties(fixture_${short_group}_${name} PROPERTIES
    PASS_REGULAR_EXPRESSION "OK:")
endfunction()

function(add_perf_fixture_negative_test group name)
  string(REGEX REPLACE "^[0-9]+_" "" short_group ${group})
  add_test(
    NAME fixture_${short_group}_${name}
    COMMAND perf --input=${CMAKE_CURRENT_SOURCE_DIR}/${name}.invalid.mlir
                 --target=${CMAKE_CURRENT_SOURCE_DIR}/target.mlir
  )
  set_tests_properties(fixture_${short_group}_${name} PROPERTIES
    WILL_FAIL TRUE)
endfunction()

add_subdirectory(0_base)
add_subdirectory(1_contract)
add_subdirectory(3_verify_reachability)
