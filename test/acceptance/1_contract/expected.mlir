// Expected output of `perf --config=config.json --dump-target`.
// Verified textually by the dump_target test (PASS_REGULAR_EXPRESSION).
//
// config.json is the architect-level graph of memory hierarchy / compute
// resources / interconnect. --dump-target projects it into the simulator-
// level executor/link vocabulary the simulator consumes directly:
//   - compute arrays          -> executor<kind = "array">
//   - interconnect dma engine -> executor "dma_read" / "dma_write",
//     with the engine's read_ports / write_ports expanded into instances
//   - memory hierarchy dram   -> dram_bus link carrying per-channel bandwidth
//
// The target carries only the resource inventory and its physical
// parameters. How each resource is modelled — the demand type it accepts, the
// placement strategy, the arbiter, and the nominal cycle formula — is keyed
// off `kind` / resource type and built into the simulator, not the target:
//   dma_read / dma_write  -> BlockQueueDemand  (whole-burst queue)   + fifo
//   array                 -> BlockFollowDemand (follow actual window)+ fifo
//   dram_bus link         -> BlockOffsetDemand (queued, non-preempt.)+ fifo
//   bank ports (sram) -> CycleDemand       (per-cycle contention)+ priority
module {
  perf.target_def @systolic_npu_v1 <
    version = 5,
    target_name = "systolic_npu_v1",
    frequency_mhz = 1000,
    memory_spaces = [
      #perf.sram<name = "spad", capacity_kib = 256, banks = 4,
                     read_ports = 4, write_ports = 4,
                     port_width_bytes = 16, row_width_bytes = 16,
                     interleave = "low">,
      #perf.sram<name = "acc", capacity_kib = 64, banks = 4,
                     read_ports = 1, write_ports = 1,
                     port_width_bytes = 64, row_width_bytes = 64,
                     interleave = "low">
    ],
    executors = [
      // The array config carries the acc read-out epilogue (scale + ReLU).
      #perf.executor<kind = "array", count = 1,
                            spaces = ["spad", "acc"],
                            config = #perf.array<topology = "systolic",
                                                 mesh_rows = 16, mesh_cols = 16,
                                                 tile_rows = 1, tile_cols = 1,
                                                 pe_latency_cycles = 1,
                                                 input_skew = true,
                                                 inter_tile_pipelined = true,
                                                 acc_read_scale_cycles = 2,
                                                 acc_read_act_cycles = 1>>,
      // dma engine config is engine-intrinsic; bandwidth is on the dram link.
      // read_ports / write_ports from the engine config were expanded into
      // the dma_read / dma_write executor families.
      #perf.executor<kind = "dma_read", count = 1,
                            spaces = ["spad", "acc"],
                            config = #perf.dma<read_ports = 1, write_ports = 1,
                                               fixed_latency_cycles = 20,
                                               max_outstanding = 8,
                                               burst_bytes = 64>>,
      #perf.executor<kind = "dma_write", count = 1,
                            spaces = ["spad", "acc"],
                            config = #perf.dma<read_ports = 1, write_ports = 1,
                                               fixed_latency_cycles = 20,
                                               max_outstanding = 8,
                                               burst_bytes = 64>>
    ],
    links = [
      // One shared DRAM channel; the simulator derives DMA transfer nominal
      // from bytes_per_cycle and resolves contention fifo (built-in rule).
      #perf.link<kind = "dram_bus", count = 1, bytes_per_cycle = 16>
    ]
  >
}
