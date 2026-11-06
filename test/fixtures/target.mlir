module attributes {perf.target = #perf.target<
  version = 3,
  target_name = "systolic_npu_v1",
  frequency_mhz = 1000,
  array = #perf.array<
    topology = "systolic",
    mesh_rows = 16, mesh_cols = 16,
    tile_rows = 1, tile_cols = 1,
    pe_latency_cycles = 1,
    input_skew = true,
    inter_tile_pipelined = true
  >,
  scratchpad = #perf.sram<
    capacity_kib = 256, banks = 4,
    read_ports = 4, write_ports = 4, port_width_bytes = 16
  >,
  accumulator = #perf.sram<
    capacity_kib = 64, banks = 4,
    read_ports = 1, write_ports = 1, port_width_bytes = 64
  >,
  dma = #perf.dma<
    bus_bytes_per_cycle = 16, fixed_latency_cycles = 20,
    max_outstanding = 8, burst_bytes = 64
  >,
  queues = #perf.queues<
    load_queue_depth = 8, store_queue_depth = 8,
    execute_queue_depth = 8, dependency_entries = 32
  >,
  resources = [
    #perf.resource_family<kind = "dma_read",  count = 1, concurrency = 1, in_order = true, placeable = true>,
    #perf.resource_family<kind = "dma_write", count = 1, concurrency = 1, in_order = true, placeable = true>,
    #perf.resource_family<kind = "array",     count = 1, concurrency = 1, in_order = true, placeable = true>,
    #perf.resource_family<kind = "dram_bus",  count = 1, concurrency = 1, in_order = true, placeable = false>
  ],
  calibration = #perf.calibration<
    extra_fill_cycles = 0, extra_drain_cycles = 0,
    extra_preload_cycles = 0, extra_writeback_cycles = 0,
    host_issue_cycles = 0
  >
>} {}
