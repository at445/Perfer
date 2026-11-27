// Provenance-only ops (perf.alloc, memref.subview, token.init, fence)
// interleaved with executor ops. The ExecutionView must skip them: only
// dma_load / array_init / compute / dma_store are indexed, in program
// order, and the subview feeding %c0 does not become a graph node.
module {
  func.func @main(%A: memref<512xi8>, %C: memref<256xi32>) {
    %free = perf.token.init

    %buf = perf.alloc : memref<512xi8, #perf.space<"spad", 0, 32>>
    %sa = memref.subview %buf[0] [256] [1]
          : memref<512xi8, #perf.space<"spad", 0, 32>>
            to memref<256xi8, strided<[1], offset: 0>, #perf.space<"spad", 0, 32>>
    %sb = memref.subview %buf[256] [256] [1]
          : memref<512xi8, #perf.space<"spad", 0, 32>>
            to memref<256xi8, strided<[1], offset: 256>, #perf.space<"spad", 0, 32>>
    %acc = perf.alloc : memref<256xi32, #perf.space<"acc", 0, 16>>

    %la = perf.dma_load %A, %buf [%free] {
      bytes = 512,
      resource = #perf.res<"dma_read", 0>
    } : (memref<512xi8>, memref<512xi8, #perf.space<"spad", 0, 32>>, !perf.token) -> !perf.token

    %i0 = perf.array_init {
      m = 16, n = 16,
      resource = #perf.res<"array", 0>
    } : () -> !perf.token

    // The compute reads through subviews: provenance-only ops sit between
    // the executor ops but are transparent to the view.
    %c0 = perf.compute %sa, %sb, %acc [%la, %i0] {
      m = 16, n = 16, k = 32,
      writeback = true,
      resource = #perf.res<"array", 0>
    } : (memref<256xi8, strided<[1], offset: 0>, #perf.space<"spad", 0, 32>>,
         memref<256xi8, strided<[1], offset: 256>, #perf.space<"spad", 0, 32>>,
         memref<256xi32, #perf.space<"acc", 0, 16>>,
         !perf.token, !perf.token) -> !perf.token

    %st = perf.dma_store %acc, %C [%c0] {
      bytes = 1024,
      resource = #perf.res<"dma_write", 0>
    } : (memref<256xi32, #perf.space<"acc", 0, 16>>, memref<256xi32>, !perf.token) -> !perf.token

    perf.fence [%st]
    return
  }
}
