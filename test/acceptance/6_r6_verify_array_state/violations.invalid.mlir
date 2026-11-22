// R6 negative fixture: PE-state provenance violations.
module {
  // compute with no array_init and no earlier compute anywhere on the
  // token chain: the PE state it accumulates onto has unknown origin.
  func.func @compute_without_init(%A: memref<512xi8>, %B: memref<512xi8>,
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

    // No array_init at all: c0's PE state has no provenance.
    %c0 = perf.compute %SA, %SB, %ACC [%la, %lb] {
      m = 16, n = 16, k = 32, writeback = true,
      resource = #perf.res<"array", 0>
    } : (memref<512xi8, #perf.space<"spad", 0, 32>>,
         memref<512xi8, #perf.space<"spad", 32, 32>>,
         memref<256xi32, #perf.space<"acc", 0, 16>>,
         !perf.token, !perf.token) -> !perf.token

    perf.fence [%c0]
    return
  }
}

// array_init with an operand reads old partial sums, but no compute on
// that array is ordered before it: the writeback it expects has not
// happened (WAR).
module {
  func.func @init_reads_unproduced_state(%ACC: memref<256xi32, #perf.space<"acc", 0, 16>>) {
    %free = perf.token.init

    // Reads old partial sums from ACC, but no compute ever produced them
    // and none is ordered before this init along the token chain.
    %i1 = perf.array_init %ACC [%free] {
      m = 16, n = 16,
      resource = #perf.res<"array", 0>
    } : (memref<256xi32, #perf.space<"acc", 0, 16>>, !perf.token) -> !perf.token

    perf.fence [%i1]
    return
  }
}
