// R3 negative fixture: schedule entries that contradict the token
// partial order (replay would deadlock).
module {
  perf.schedule {
    entries = [
      #perf.schedule_entry<#perf.res<"dma_read", 0>, ["load_b", "load_a"]>
    ]
  }

  func.func @main(%A: memref<512xi8>, %B: memref<512xi8>,
                  %SA: memref<512xi8, #perf.space<"spad", 0, 32>>,
                  %SB: memref<512xi8, #perf.space<"spad", 32, 32>>) {
    %free = perf.token.init

    // store depends on load_a via token, but the schedule orders store
    // before load_a on the same resource.
    %la = perf.dma_load %A, %SA [%free] {
      bytes = 512,
      resource = #perf.res<"dma_read", 0>,
      sched_name = "load_a"
    } : (memref<512xi8>, memref<512xi8, #perf.space<"spad", 0, 32>>, !perf.token) -> !perf.token

    %lb = perf.dma_load %B, %SB [%la] {
      bytes = 512,
      resource = #perf.res<"dma_read", 0>,
      sched_name = "load_b"
    } : (memref<512xi8>, memref<512xi8, #perf.space<"spad", 32, 32>>, !perf.token) -> !perf.token

    perf.fence [%lb]
    return
  }

}
