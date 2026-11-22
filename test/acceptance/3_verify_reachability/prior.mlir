// Simulation priors for the systolic_npu_v1 machine family. These are
// calibration corrections produced by alignment against RTL ground truth;
// a DSE sweep shares this file unchanged across all design points.

perf.prior_def @systolic_npu_base #perf.prior<
  version = 4,
  target_name = "systolic_npu_base",
  calibration = #perf.calibration<
    extra_fill_cycles = 0, extra_drain_cycles = 0,
    extra_preload_cycles = 0, extra_writeback_cycles = 0,
    host_issue_cycles = 0
  >
>
