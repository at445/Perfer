// Design point in the DSE engineer's vocabulary: physical structure
// parameters. The compiler-simulator contract (executor instances, links)
// is projected from this; see `perf --dump-contract`.
perf.target_def @systolic_npu_v1 #perf.design<
  version = 5,
  target_name = "systolic_npu_v1",
  frequency_mhz = 1000,
  memory_spaces = [
    #perf.memspace<
      name = "spad",
      capacity_kib = 256, banks = 4,
      read_ports = 4, write_ports = 4,
      port_width_bytes = 16, row_width_bytes = 16
    >,
    #perf.memspace<
      name = "acc",
      capacity_kib = 64, banks = 4,
      read_ports = 1, write_ports = 1,
      port_width_bytes = 64, row_width_bytes = 64
    >
  ],
  dma_read_ports = 1,
  dma_write_ports = 1,
  arrays = 1,
  dma_channel = #perf.dma<
    bus_bytes_per_cycle = 16, fixed_latency_cycles = 20,
    max_outstanding = 8, burst_bytes = 64
  >,
  array_uarch = #perf.array<
    topology = "systolic",
    mesh_rows = 16, mesh_cols = 16,
    tile_rows = 1, tile_cols = 1,
    pe_latency_cycles = 1,
    input_skew = true,
    inter_tile_pipelined = true
  >,
  share_dram_bus = true
>
