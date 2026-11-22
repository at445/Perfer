module {
  func.func @bad(%A: memref<512xi8>, %B: memref<512xi8, #perf.space<"l2", 0, 32>>) {
    %t = perf.token.init
    // R1: memory space "l2" is not declared in the design.
    %l = perf.dma_load %A, %B [%t] {
      bytes = 512,
      resource = #perf.res<"dma_read", 0>
    } : (memref<512xi8>, memref<512xi8, #perf.space<"l2", 0, 32>>, !perf.token) -> !perf.token
    return
  }
}
