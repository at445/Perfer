// Shared DMA channel parameters; referenced by both DMA families so the
// read/write channels stay consistent by construction.
#dma_channel = #perf.dma<
  bus_bytes_per_cycle = 16, fixed_latency_cycles = 20,
  max_outstanding = 8, burst_bytes = 64
>

// Systolic array microarchitecture.
#systolic_array = #perf.array<
  topology = "systolic",
  mesh_rows = 16, mesh_cols = 16,
  tile_rows = 1, tile_cols = 1,
  pe_latency_cycles = 1,
  input_skew = true,
  inter_tile_pipelined = true
>

perf.target_def @systolic_npu_v1 #perf.target<
  version = 4,
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
  executors = [
    #perf.executor_family<
      kind = "dma_read", count = 1, in_order = true,
      spaces = ["spad", "acc"], config = #dma_channel
    >,
    #perf.executor_family<
      kind = "dma_write", count = 1, in_order = true,
      spaces = ["spad", "acc"], config = #dma_channel
    >,
    #perf.executor_family<
      kind = "array", count = 1, in_order = true,
      spaces = ["spad", "acc"], config = #systolic_array
    >
  ],
  links = [
    #perf.link<kind = "dram_bus", count = 1>
  ],
  calibration = #perf.calibration<
    extra_fill_cycles = 0, extra_drain_cycles = 0,
    extra_preload_cycles = 0, extra_writeback_cycles = 0,
    host_issue_cycles = 0
  >
>
