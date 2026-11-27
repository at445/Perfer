// Shared prior used by all TargetModel tests.
module attributes {perf.prior = #perf.prior<
  version = 4,
  target_name = "test_prior",
  calibration = #perf.calibration<
    extra_fill_cycles = 0, extra_drain_cycles = 0,
    extra_preload_cycles = 0, extra_writeback_cycles = 0,
    host_issue_cycles = 0
  >
>} {}
