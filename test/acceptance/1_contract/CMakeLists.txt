# ---------------------------------------------------------------------------
# Contract projection (--dump-contract): the projected compiler-simulator
# contract must contain the complete executor/link vocabulary of the design.
# expected.mlir documents the full expected output for reference.
# ---------------------------------------------------------------------------

add_test(
  NAME contract_dump
  COMMAND perf --input=${CMAKE_CURRENT_SOURCE_DIR}/../0_base/input.mlir
               --target=${CMAKE_CURRENT_SOURCE_DIR}/target.mlir
               --dump-contract
)
set_tests_properties(contract_dump PROPERTIES
  PASS_REGULAR_EXPRESSION
    "perf.contract_def @systolic_npu_v1.*executor_family<kind = \"dma_read\".*executor_family<kind = \"dma_write\".*executor_family<kind = \"array\".*link<kind = \"dram_bus\"")
