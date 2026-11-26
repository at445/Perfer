// R1 negative fixture collection. Each function violates exactly one R1 rule;
// the verifier must reject the module (any violation fails the run).
module {

  // R1: "dma_rade" is not an executor kind in the contract.
  func.func @unknown_kind(%A: memref<512xi8>, %B: memref<512xi8, #perf.space<"spad", 0, 32>>) {
    %t = perf.token.init
    %l = perf.dma_load %A, %B [%t] {
      bytes = 512,
      resource = #perf.res<"dma_rade", 0>
    } : (memref<512xi8>, memref<512xi8, #perf.space<"spad", 0, 32>>, !perf.token) -> !perf.token
    return
  }

  // R1: the design has dma_read_ports = 1, so dma_read#7 does not exist.
  func.func @index_out_of_range(%A: memref<512xi8>, %B: memref<512xi8, #perf.space<"spad", 0, 32>>) {
    %t = perf.token.init
    %l = perf.dma_load %A, %B [%t] {
      bytes = 512,
      resource = #perf.res<"dma_read", 7>
    } : (memref<512xi8>, memref<512xi8, #perf.space<"spad", 0, 32>>, !perf.token) -> !perf.token
    return
  }

  // R1: memory space "l2" is not declared in the design.
  func.func @unknown_space(%A: memref<512xi8>, %B: memref<512xi8, #perf.space<"l2", 0, 32>>) {
    %t = perf.token.init
    %l = perf.dma_load %A, %B [%t] {
      bytes = 512,
      resource = #perf.res<"dma_read", 0>
    } : (memref<512xi8>, memref<512xi8, #perf.space<"l2", 0, 32>>, !perf.token) -> !perf.token
    return
  }

  // R1: "dram_bus" is a link (internal contention channel), not an
  // executor; links cannot be referenced by IR placement.
  func.func @link_as_placement(%A: memref<512xi8>, %B: memref<512xi8, #perf.space<"spad", 0, 32>>) {
    %t = perf.token.init
    %l = perf.dma_load %A, %B [%t] {
      bytes = 512,
      resource = #perf.res<"dram_bus", 0>
    } : (memref<512xi8>, memref<512xi8, #perf.space<"spad", 0, 32>>, !perf.token) -> !perf.token
    return
  }

  // R1: the access window runs past the space's physical rows. spad has
  // banks=4, row_width_bytes=16, capacity=256 KiB -> 4096 rows/bank, 16384
  // rows total; [16000, 16000+1000) = 17000 > 16384, landing on nonexistent
  // banks.
  func.func @row_out_of_bounds(%A: memref<512xi8>, %B: memref<512xi8, #perf.space<"spad", 16000, 1000>>) {
    %t = perf.token.init
    %l = perf.dma_load %A, %B [%t] {
      bytes = 512,
      resource = #perf.res<"dma_read", 0>
    } : (memref<512xi8>, memref<512xi8, #perf.space<"spad", 16000, 1000>>, !perf.token) -> !perf.token
    return
  }
}
