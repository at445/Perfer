module {
  func.func @bad(%A: memref<512xi8>, %B: memref<512xi8, #perf.space<"spad", 0, 32>>) {
    %t = perf.token.init
    // R1: "dma_rade" is not an executor kind in the contract.
    %l = perf.dma_load %A, %B [%t] {
      bytes = 512,
      resource = #perf.res<"dma_rade", 0>
    } : (memref<512xi8>, memref<512xi8, #perf.space<"spad", 0, 32>>, !perf.token) -> !perf.token
    return
  }
}
