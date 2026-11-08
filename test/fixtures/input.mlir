module {

  func.func @matmul_impl(%A: memref<512xi8>, %B: memref<512xi8>, %C: memref<256xi32>) {
    %free = perf.token.init

    // buffers
    %spad_a0 = perf.alloc : memref<512xi8, #perf.space<"spad", 0>>
    %spad_b  = perf.alloc : memref<512xi8, #perf.space<"spad", 2>>
    %acc_c0  = perf.alloc : memref<256xi32, #perf.space<"acc", 0>>

    // DMA load A and B into spad
    %la = perf.dma_load %A, %spad_a0 [%free] {
      bytes = 512,
      resource = #perf.res<"dma_read", 0>
    } : (memref<512xi8>, memref<512xi8, #perf.space<"spad", 0>>, !perf.token) -> !perf.token

    %lb = perf.dma_load %B, %spad_b [%free] {
      bytes = 512,
      resource = #perf.res<"dma_read", 0>
    } : (memref<512xi8>, memref<512xi8, #perf.space<"spad", 2>>, !perf.token) -> !perf.token

    // preload C init
    %p0 = perf.preload %acc_c0 {
      role = "output", m = 16, n = 16, init = "zero",
      resource = #perf.res<"array", 0>
    } : (memref<256xi32, #perf.space<"acc", 0>>) -> !perf.token

    // compute
    %c0 = perf.compute %spad_a0, %spad_b, %acc_c0 [%la, %lb, %p0] {
      m = 16, n = 16, k = 32,
      accumulate = false, writeback = true,
      resource = #perf.res<"array", 10>
    } : (memref<512xi8, #perf.space<"spad", 0>>,
         memref<512xi8, #perf.space<"spad", 2>>,
         memref<256xi32, #perf.space<"acc", 0>>,
         !perf.token, !perf.token, !perf.token) -> !perf.token

    // store result
    %st0 = perf.dma_store %acc_c0, %C [%c0] {
      bytes = 1024,
      resource = #perf.res<"dma_write", 0>
    } : (memref<256xi32, #perf.space<"acc", 0>>, memref<256xi32>, !perf.token) -> !perf.token

    perf.wait [%st0]
    return
  }
}
