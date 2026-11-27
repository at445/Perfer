// Multi-instance expansion: two array instances and two read/write DMA
// channels sharing two DRAM channels. Exercises count-expansion ordering
// and per-instance naming in the dump.
module attributes {perf.target = #perf.target<
  version = 4,
  target_name = "multi_array",
  frequency_mhz = 1000,
  memory_spaces = [
    #perf.sram<
      name = "spad",
      capacity_kib = 256, banks = 4,
      read_ports = 4, write_ports = 4,
      port_width_bytes = 16, row_width_bytes = 16, interleave = "low"
    >,
    #perf.sram<
      name = "acc",
      capacity_kib = 64, banks = 4,
      read_ports = 1, write_ports = 1,
      port_width_bytes = 64, row_width_bytes = 64, interleave = "low"
    >
  ],
  executors = [
    #perf.executor<
      kind = "array", count = 2,
      spaces = ["spad", "acc"],
      config = #perf.array<topology = "systolic",
                           mesh_rows = 16, mesh_cols = 16,
                           tile_rows = 2, tile_cols = 2,
                           pe_latency_cycles = 1,
                           input_skew = true,
                           inter_tile_pipelined = true,
                           acc_read_scale_cycles = 2,
                           acc_read_act_cycles = 1>
    >,
    #perf.executor<
      kind = "dma_read", count = 2,
      spaces = ["spad", "acc"],
      config = #perf.dma<read_ports = 2, write_ports = 0,
                         fixed_latency_cycles = 20,
                         max_outstanding = 8, burst_bytes = 64>
    >,
    #perf.executor<
      kind = "dma_write", count = 2,
      spaces = ["spad", "acc"],
      config = #perf.dma<read_ports = 0, write_ports = 2,
                         fixed_latency_cycles = 20,
                         max_outstanding = 8, burst_bytes = 64>
    >
  ],
  links = [
    #perf.link<kind = "dram_bus", count = 2, bytes_per_cycle = 16>
  ]
>} {}
