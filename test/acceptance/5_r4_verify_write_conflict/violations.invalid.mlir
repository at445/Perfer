// R4 negative fixture: two unordered accesses conflict on the same buffer,
// at least one of which is a write.
module {
  func.func @main(%A: memref<512xi8>, %B: memref<512xi8>,
                  %SA: memref<512xi8, #perf.space<"spad", 0, 32>>,
                  %C: memref<256xi32>) {
    %free = perf.token.init

    // Write to %SA via dma_load's dst...
    %la = perf.dma_load %A, %SA [%free] {
      bytes = 512,
      resource = #perf.res<"dma_read", 0>
    } : (memref<512xi8>, memref<512xi8, #perf.space<"spad", 0, 32>>, !perf.token) -> !perf.token

    // ...and an unordered second write into the same buffer via dma_store's
    // src (read) is NOT a conflict by itself; the conflict below is the
    // unordered WRITE to the same buffer by another load.
    %lb = perf.dma_load %B, %SA [%free] {
      bytes = 512,
      resource = #perf.res<"dma_read", 0>
    } : (memref<512xi8>, memref<512xi8, #perf.space<"spad", 0, 32>>, !perf.token) -> !perf.token

    perf.fence [%la, %lb]
    return
  }
}

// Two computes on different inputs but the same accumulator, with no token
// between them: both write the accumulator, an unordered write conflict.
module {
  func.func @shared_accumulator(%A: memref<512xi8>, %B: memref<512xi8>,
                                %SA0: memref<512xi8, #perf.space<"spad", 0, 32>>,
                                %SA1: memref<512xi8, #perf.space<"spad", 32, 32>>,
                                %SB: memref<512xi8, #perf.space<"spad", 64, 32>>,
                                %ACC: memref<256xi32, #perf.space<"acc", 0, 16>>) {
    %free = perf.token.init

    %la0 = perf.dma_load %A, %SA0 [%free] {
      bytes = 512, resource = #perf.res<"dma_read", 0>
    } : (memref<512xi8>, memref<512xi8, #perf.space<"spad", 0, 32>>, !perf.token) -> !perf.token

    %la1 = perf.dma_load %A, %SA1 [%free] {
      bytes = 512, resource = #perf.res<"dma_read", 0>
    } : (memref<512xi8>, memref<512xi8, #perf.space<"spad", 32, 32>>, !perf.token) -> !perf.token

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

    %c1 = perf.compute %SA1, %SB, %ACC [%la1, %lb, %i0] {
      m = 16, n = 16, k = 32, writeback = true,
      resource = #perf.res<"array", 0>
    } : (memref<512xi8, #perf.space<"spad", 32, 32>>,
         memref<512xi8, #perf.space<"spad", 64, 32>>,
         memref<256xi32, #perf.space<"acc", 0, 16>>,
         !perf.token, !perf.token, !perf.token) -> !perf.token

    perf.fence [%c0, %c1]
    return
  }
}

// Buffer rotation missing the WAR token: the next rotation reloads buffer 0
// while the previous compute on it may still be reading it. The load's write
// conflicts with the compute's read, unordered.
module {
  func.func @rotation_missing_war(%A: memref<1024xi8>, %B: memref<512xi8>,
                                  %SA0: memref<512xi8, #perf.space<"spad", 0, 32>>,
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

    // Reloads buffer 0 without waiting for c0: unordered WAR conflict.
    %la1 = perf.dma_load %A, %SA0 [%free] {
      bytes = 512, resource = #perf.res<"dma_read", 0>
    } : (memref<1024xi8>, memref<512xi8, #perf.space<"spad", 0, 32>>, !perf.token) -> !perf.token

    perf.fence [%c0, %la1]
    return
  }
}

// Premature drain: dma_store reads the accumulator while a compute is still
// accumulating into it. The store's read conflicts with the compute's
// write, unordered.
module {
  func.func @premature_store(%A: memref<512xi8>, %B: memref<512xi8>, %C: memref<256xi32>,
                             %SA: memref<512xi8, #perf.space<"spad", 0, 32>>,
                             %SB: memref<512xi8, #perf.space<"spad", 64, 32>>,
                             %ACC: memref<256xi32, #perf.space<"acc", 0, 16>>) {
    %free = perf.token.init

    %la = perf.dma_load %A, %SA [%free] {
      bytes = 512, resource = #perf.res<"dma_read", 0>
    } : (memref<512xi8>, memref<512xi8, #perf.space<"spad", 0, 32>>, !perf.token) -> !perf.token

    %lb = perf.dma_load %B, %SB [%free] {
      bytes = 512, resource = #perf.res<"dma_read", 0>
    } : (memref<512xi8>, memref<512xi8, #perf.space<"spad", 64, 32>>, !perf.token) -> !perf.token

    %i0 = perf.array_init {
      m = 16, n = 16, resource = #perf.res<"array", 0>
    } : () -> !perf.token

    %c0 = perf.compute %SA, %SB, %ACC [%la, %lb, %i0] {
      m = 16, n = 16, k = 32, writeback = true,
      resource = #perf.res<"array", 0>
    } : (memref<512xi8, #perf.space<"spad", 0, 32>>,
         memref<512xi8, #perf.space<"spad", 64, 32>>,
         memref<256xi32, #perf.space<"acc", 0, 16>>,
         !perf.token, !perf.token, !perf.token) -> !perf.token

    // Drains the accumulator without waiting for c0: unordered conflict.
    %st = perf.dma_store %ACC, %C [%free] {
      bytes = 1024, resource = #perf.res<"dma_write", 0>
    } : (memref<256xi32, #perf.space<"acc", 0, 16>>, memref<256xi32>, !perf.token) -> !perf.token

    perf.fence [%c0, %st]
    return
  }
}

// array_init with an operand reads the old partial sums while a previous
// compute's writeback into the same accumulator has not completed. The
// init's read conflicts with the compute's write, unordered.
module {
  func.func @init_races_writeback(%A: memref<512xi8>, %B: memref<512xi8>,
                                  %SA: memref<512xi8, #perf.space<"spad", 0, 32>>,
                                  %SB: memref<512xi8, #perf.space<"spad", 64, 32>>,
                                  %ACC: memref<256xi32, #perf.space<"acc", 0, 16>>) {
    %free = perf.token.init

    %la = perf.dma_load %A, %SA [%free] {
      bytes = 512, resource = #perf.res<"dma_read", 0>
    } : (memref<512xi8>, memref<512xi8, #perf.space<"spad", 0, 32>>, !perf.token) -> !perf.token

    %lb = perf.dma_load %B, %SB [%free] {
      bytes = 512, resource = #perf.res<"dma_read", 0>
    } : (memref<512xi8>, memref<512xi8, #perf.space<"spad", 64, 32>>, !perf.token) -> !perf.token

    %i0 = perf.array_init {
      m = 16, n = 16, resource = #perf.res<"array", 0>
    } : () -> !perf.token

    %c0 = perf.compute %SA, %SB, %ACC [%la, %lb, %i0] {
      m = 16, n = 16, k = 32, writeback = true,
      resource = #perf.res<"array", 0>
    } : (memref<512xi8, #perf.space<"spad", 0, 32>>,
         memref<512xi8, #perf.space<"spad", 64, 32>>,
         memref<256xi32, #perf.space<"acc", 0, 16>>,
         !perf.token, !perf.token, !perf.token) -> !perf.token

    // Reads ACC without waiting for c0's writeback: unordered conflict.
    %i1 = perf.array_init %ACC [%free] {
      m = 16, n = 16,
      resource = #perf.res<"array", 0>
    } : (memref<256xi32, #perf.space<"acc", 0, 16>>, !perf.token) -> !perf.token

    perf.fence [%c0, %i1]
    return
  }
}
