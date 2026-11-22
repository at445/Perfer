// R4 positive fixture: write conflicts are absent because conflicting
// accesses are ordered by the token partial order; read-read pairs are a
// performance dimension, not an error.
module {
  func.func @main(%A: memref<512xi8>, %B: memref<512xi8>,
                  %SA: memref<512xi8, #perf.space<"spad", 0, 32>>,
                  %SB: memref<512xi8, #perf.space<"spad", 32, 32>>) {
    %free = perf.token.init

    // WAR on %SA: the load reads %SA before the store overwrites it; the
    // token makes the ordering explicit, so the pair is legal.
    %la = perf.dma_load %A, %SA [%free] {
      bytes = 512,
      resource = #perf.res<"dma_read", 0>
    } : (memref<512xi8>, memref<512xi8, #perf.space<"spad", 0, 32>>, !perf.token) -> !perf.token

    %st = perf.dma_store %SA, %A [%la] {
      bytes = 512,
      resource = #perf.res<"dma_write", 0>
    } : (memref<512xi8, #perf.space<"spad", 0, 32>>, memref<512xi8>, !perf.token) -> !perf.token

    // An independent load into a DIFFERENT buffer: no conflict at all.
    %lb = perf.dma_load %B, %SB [%free] {
      bytes = 512,
      resource = #perf.res<"dma_read", 0>
    } : (memref<512xi8>, memref<512xi8, #perf.space<"spad", 32, 32>>, !perf.token) -> !perf.token

    perf.fence [%st, %lb]
    return
  }
}

// Double-buffered pipeline: every conflict is token-ordered (WAR between
// load and compute on the same buffer, RAW between compute and store),
// so R4 must pass even with cross-resource dataflow.
module {
  func.func @double_buffer(%A: memref<1024xi8>, %B: memref<512xi8>, %C: memref<256xi32>,
                           %SA0: memref<512xi8, #perf.space<"spad", 0, 32>>,
                           %SA1: memref<512xi8, #perf.space<"spad", 32, 32>>,
                           %SB: memref<512xi8, #perf.space<"spad", 64, 32>>,
                           %ACC: memref<256xi32, #perf.space<"acc", 0, 16>>) {
    %free = perf.token.init

    %la0 = perf.dma_load %A, %SA0 [%free] {
      bytes = 512, resource = #perf.res<"dma_read", 0>
    } : (memref<1024xi8>, memref<512xi8, #perf.space<"spad", 0, 32>>, !perf.token) -> !perf.token

    %lb = perf.dma_load %B, %SB [%free] {
      bytes = 512, resource = #perf.res<"dma_read", 0>
    } : (memref<512xi8>, memref<512xi8, #perf.space<"spad", 64, 32>>, !perf.token) -> !perf.token

    %i0 = perf.array_init {
      m = 16, n = 16, resource = #perf.res<"array", 0>
    } : () -> !perf.token

    %c0 = perf.compute %SA0, %SB, %ACC [%la0, %lb, %i0] {
      m = 16, n = 16, k = 32, writeback = true,
      resource = #perf.res<"array", 0>
    } : (memref<512xi8, #perf.space<"spad", 0, 32>>,
         memref<512xi8, #perf.space<"spad", 64, 32>>,
         memref<256xi32, #perf.space<"acc", 0, 16>>,
         !perf.token, !perf.token, !perf.token) -> !perf.token

    %la1 = perf.dma_load %A, %SA1 [%free] {
      bytes = 512, resource = #perf.res<"dma_read", 0>
    } : (memref<1024xi8>, memref<512xi8, #perf.space<"spad", 32, 32>>, !perf.token) -> !perf.token

    %c1 = perf.compute %SA1, %SB, %ACC [%la1, %lb, %c0] {
      m = 16, n = 16, k = 32, writeback = true,
      resource = #perf.res<"array", 0>
    } : (memref<512xi8, #perf.space<"spad", 32, 32>>,
         memref<512xi8, #perf.space<"spad", 64, 32>>,
         memref<256xi32, #perf.space<"acc", 0, 16>>,
         !perf.token, !perf.token, !perf.token) -> !perf.token

    %st = perf.dma_store %ACC, %C [%c1] {
      bytes = 1024, resource = #perf.res<"dma_write", 0>
    } : (memref<256xi32, #perf.space<"acc", 0, 16>>, memref<256xi32>, !perf.token) -> !perf.token

    perf.fence [%st]
    return
  }
}
