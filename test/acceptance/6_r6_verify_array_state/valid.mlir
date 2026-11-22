// R6 positive fixture: every compute's PE state traces back to array_init
// or an earlier compute along the token chain.
module {
  func.func @main(%A: memref<512xi8>, %B: memref<512xi8>,
                  %SA: memref<512xi8, #perf.space<"spad", 0, 32>>,
                  %SB: memref<512xi8, #perf.space<"spad", 32, 32>>,
                  %ACC: memref<256xi32, #perf.space<"acc", 0, 16>>) {
    %free = perf.token.init

    %la = perf.dma_load %A, %SA [%free] {
      bytes = 512, resource = #perf.res<"dma_read", 0>
    } : (memref<512xi8>, memref<512xi8, #perf.space<"spad", 0, 32>>, !perf.token) -> !perf.token

    %lb = perf.dma_load %B, %SB [%free] {
      bytes = 512, resource = #perf.res<"dma_read", 0>
    } : (memref<512xi8>, memref<512xi8, #perf.space<"spad", 32, 32>>, !perf.token) -> !perf.token

    // PE state origin: zero-fill init.
    %i0 = perf.array_init {
      m = 16, n = 16, resource = #perf.res<"array", 0>
    } : () -> !perf.token

    // First tile: state comes from i0.
    %c0 = perf.compute %SA, %SB, %ACC [%la, %lb, %i0] {
      m = 16, n = 16, k = 32, writeback = false,
      resource = #perf.res<"array", 0>
    } : (memref<512xi8, #perf.space<"spad", 0, 32>>,
         memref<512xi8, #perf.space<"spad", 32, 32>>,
         memref<256xi32, #perf.space<"acc", 0, 16>>,
         !perf.token, !perf.token, !perf.token) -> !perf.token

    // Second tile accumulates onto c0's state: provenance is c0.
    %c1 = perf.compute %SA, %SB, %ACC [%la, %lb, %c0] {
      m = 16, n = 16, k = 32, writeback = true,
      resource = #perf.res<"array", 0>
    } : (memref<512xi8, #perf.space<"spad", 0, 32>>,
         memref<512xi8, #perf.space<"spad", 32, 32>>,
         memref<256xi32, #perf.space<"acc", 0, 16>>,
         !perf.token, !perf.token, !perf.token) -> !perf.token

    perf.fence [%c1]
    return
  }
}
