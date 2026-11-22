// R1 positive fixture collection. Every function must pass reachability
// verification against test/fixtures/target.mlir.
//
// Coverage: one valid form per op kind, plus boundary cases that look
// similar to violations but are legal.
module {

  // Canonical matmul: dma_load / array_init (no operand) / compute / dma_store.
  func.func @matmul(%A: memref<512xi8>, %B: memref<512xi8>, %C: memref<256xi32>) {
    %free = perf.token.init

    %spad_a = perf.alloc : memref<512xi8, #perf.space<"spad", 0, 32>>
    %spad_b = perf.alloc : memref<512xi8, #perf.space<"spad", 32, 32>>
    %acc_c  = perf.alloc : memref<256xi32, #perf.space<"acc", 0, 16>>

    %la = perf.dma_load %A, %spad_a [%free] {
      bytes = 512,
      resource = #perf.res<"dma_read", 0>
    } : (memref<512xi8>, memref<512xi8, #perf.space<"spad", 0, 32>>, !perf.token) -> !perf.token

    %lb = perf.dma_load %B, %spad_b [%free] {
      bytes = 512,
      resource = #perf.res<"dma_read", 0>
    } : (memref<512xi8>, memref<512xi8, #perf.space<"spad", 32, 32>>, !perf.token) -> !perf.token

    %i0 = perf.array_init {
      m = 16, n = 16,
      resource = #perf.res<"array", 0>
    } : () -> !perf.token

    %c0 = perf.compute %spad_a, %spad_b, %acc_c [%la, %lb, %i0] {
      m = 16, n = 16, k = 32,
      writeback = true,
      resource = #perf.res<"array", 0>
    } : (memref<512xi8, #perf.space<"spad", 0, 32>>,
         memref<512xi8, #perf.space<"spad", 32, 32>>,
         memref<256xi32, #perf.space<"acc", 0, 16>>,
         !perf.token, !perf.token, !perf.token) -> !perf.token

    %s0 = perf.dma_store %acc_c, %C [%c0] {
      bytes = 1024,
      resource = #perf.res<"dma_write", 0>
    } : (memref<256xi32, #perf.space<"acc", 0, 16>>, memref<256xi32>, !perf.token) -> !perf.token

    perf.wait [%s0]
    return
  }

  // array_init with an operand: reads old partial sums from the acc space.
  func.func @array_init_with_operand(%acc: memref<256xi32, #perf.space<"acc", 0, 16>>) {
    %t = perf.token.init
    %i0 = perf.array_init %acc [%t] {
      m = 16, n = 16,
      resource = #perf.res<"array", 0>
    } : (memref<256xi32, #perf.space<"acc", 0, 16>>, !perf.token) -> !perf.token
    return
  }

  // dma_store on its own: bridges the on-chip src and the host-side dst.
  func.func @dma_store_host_dst(%C: memref<256xi32>,
                                %acc: memref<256xi32, #perf.space<"acc", 0, 16>>) {
    %t = perf.token.init
    %s0 = perf.dma_store %acc, %C [%t] {
      bytes = 1024,
      resource = #perf.res<"dma_write", 0>
    } : (memref<256xi32, #perf.space<"acc", 0, 16>>, memref<256xi32>, !perf.token) -> !perf.token
    return
  }

  // Boundary: index 0 is the only valid instance of a single-port family.
  func.func @boundary_index_zero(%A: memref<512xi8>, %B: memref<512xi8, #perf.space<"spad", 0, 32>>) {
    %t = perf.token.init
    %l = perf.dma_load %A, %B [%t] {
      bytes = 512,
      resource = #perf.res<"dma_read", 0>
    } : (memref<512xi8>, memref<512xi8, #perf.space<"spad", 0, 32>>, !perf.token) -> !perf.token
    return
  }
}
