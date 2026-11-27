// Two dma_loads on the same dma_read channel with explicit order
// annotations that invert program order: order=1 (lb) precedes order=2 (la).
module {
  func.func @t(%A: memref<512xi8>, %B: memref<512xi8>,
               %SA: memref<512xi8, #perf.space<"spad", 0, 32>>,
               %SB: memref<512xi8, #perf.space<"spad", 32, 32>>) {
    %free = perf.token.init
    %la = perf.dma_load %A, %SA [%free] {
      bytes = 512, resource = #perf.res<"dma_read", 0>, order = 2
    } : (memref<512xi8>, memref<512xi8, #perf.space<"spad", 0, 32>>, !perf.token) -> !perf.token
    %lb = perf.dma_load %B, %SB [%free] {
      bytes = 512, resource = #perf.res<"dma_read", 0>, order = 1
    } : (memref<512xi8>, memref<512xi8, #perf.space<"spad", 32, 32>>, !perf.token) -> !perf.token
    return
  }
}
