module {
  func.func @bad(%A: memref<512xi8>,
                 %B: memref<512xi8, #perf.space<"spad", 0, 32>>,
                 %C: memref<256xi32, #perf.space<"acc", 0, 16>>) {
    %t = perf.token.init
    // R1: the array is not a DMA bridge; it cannot touch host memory (%A).
    %c0 = perf.compute %A, %B, %C [%t] {
      m = 16, n = 16, k = 32,
      writeback = true,
      resource = #perf.res<"array", 0>
    } : (memref<512xi8>,
         memref<512xi8, #perf.space<"spad", 0, 32>>,
         memref<256xi32, #perf.space<"acc", 0, 16>>,
         !perf.token) -> !perf.token
    return
  }
}
