# ---------------------------------------------------------------------------
# Target projection (--dump-target): config.json (the architect-level graph of
# memory hierarchy / compute resources / interconnect) is projected into the
# simulator-level executor/link vocabulary the simulator consumes directly.
# expected.mlir documents the full expected --dump-target output.
# ---------------------------------------------------------------------------

add_test(
  NAME dump_target
  COMMAND perf --config=${CMAKE_CURRENT_SOURCE_DIR}/config.json
               --dump-target
)
set_tests_properties(dump_target PROPERTIES
  PASS_REGULAR_EXPRESSION
    "perf.target_def @systolic_npu_v1.*executor<kind = \"array\".*executor<kind = \"dma_read\".*executor<kind = \"dma_write\".*link<kind = \"dram_bus\".*bytes_per_cycle = 16"
)
