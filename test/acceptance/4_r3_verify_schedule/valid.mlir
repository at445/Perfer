// R3 positive fixture: perf.schedule entries consistent with the token
// partial order.
module {
  perf.schedule {
    entries = [
      #perf.schedule_entry<#perf.res<"dma_read", 0>, ["load_b", "load_a"]>,
      #perf.schedule_entry<#perf.res<"dma_write", 0>, ["store_a"]>
    ]
  }

  func.func @main(%A: memref<512xi8>, %B: memref<512xi8>,
                  %SA: memref<512xi8, #perf.space<"spad", 0, 32>>,
                  %SB: memref<512xi8, #perf.space<"spad", 32, 32>>) {
    %free = perf.token.init

    // Two independent DMA loads on the same channel. The schedule swaps
    // their issue order relative to program order; legal because neither
    // depends on the other.
    %la = perf.dma_load %A, %SA [%free] {
      bytes = 512,
      resource = #perf.res<"dma_read", 0>,
      sched_name = "load_a"
    } : (memref<512xi8>, memref<512xi8, #perf.space<"spad", 0, 32>>, !perf.token) -> !perf.token

    %lb = perf.dma_load %B, %SB [%free] {
      bytes = 512,
      resource = #perf.res<"dma_read", 0>,
      sched_name = "load_b"
    } : (memref<512xi8>, memref<512xi8, #perf.space<"spad", 32, 32>>, !perf.token) -> !perf.token

    // Dependent pair: store depends on load; schedule respects the order.
    %st = perf.dma_store %SA, %A [%la] {
      bytes = 512,
      resource = #perf.res<"dma_write", 0>,
      sched_name = "store_a"
    } : (memref<512xi8, #perf.space<"spad", 0, 32>>, memref<512xi8>, !perf.token) -> !perf.token

    perf.fence [%lb, %st]
    return
  }

}
