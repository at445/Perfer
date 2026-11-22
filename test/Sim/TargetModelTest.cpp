// TargetModelTest.cpp - Unit tests for TargetModel.

#include "perf/Sim/TargetModel.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "gtest/gtest.h"

using namespace mlir;
using namespace perf;

class TargetModelTest : public ::testing::Test {
protected:
  void SetUp() override { context.loadDialect<PerfDialect>(); }

  MLIRContext context;

  ModuleOp parseModule(StringRef text) {
    return parseSourceString<ModuleOp>(text, &context).get();
  }

  TargetAttr getTargetAttr(ModuleOp mod) {
    return llvm::dyn_cast_if_present<TargetAttr>(
        mod->getAttr("perf.target"));
  }

  FailureOr<TargetModel> buildFromText(StringRef text) {
    auto mod = parseModule(text);
    if (!mod)
      return failure();
    auto attr = getTargetAttr(mod);
    if (!attr)
      return failure();
    return TargetModel::build(attr);
  }
};

// Schema v4 baseline config: per-family configs carried inline.
static constexpr const char *kV4Config = R"mlir(
module attributes {perf.target = #perf.target<
  version = 4,
  target_name = "test_npu",
  frequency_mhz = 1000,
  memory_spaces = [
    #perf.memspace<
      name = "spad",
      capacity_kib = 256, banks = 4,
      read_ports = 4, write_ports = 4,
      port_width_bytes = 16, row_width_bytes = 16
    >,
    #perf.memspace<
      name = "acc",
      capacity_kib = 64, banks = 4,
      read_ports = 1, write_ports = 1,
      port_width_bytes = 64, row_width_bytes = 64
    >
  ],
  executors = [
    #perf.executor_family<
      kind = "dma_read", count = 1, in_order = true, spaces = ["spad", "acc"],
      config = #perf.dma<bus_bytes_per_cycle = 16, fixed_latency_cycles = 20,
                         max_outstanding = 8, burst_bytes = 64>
    >,
    #perf.executor_family<
      kind = "dma_write", count = 1, in_order = true, spaces = ["spad", "acc"],
      config = #perf.dma<bus_bytes_per_cycle = 16, fixed_latency_cycles = 20,
                         max_outstanding = 8, burst_bytes = 64>
    >,
    #perf.executor_family<
      kind = "array", count = 1, in_order = true, spaces = ["spad", "acc"],
      config = #perf.array<topology = "systolic",
                           mesh_rows = 16, mesh_cols = 16,
                           tile_rows = 1, tile_cols = 1,
                           pe_latency_cycles = 1,
                           input_skew = true,
                           inter_tile_pipelined = true>
    >
  ],
  links = [
    #perf.link<kind = "dram_bus", count = 1>
  ],
  calibration = #perf.calibration<
    extra_fill_cycles = 0, extra_drain_cycles = 0,
    extra_preload_cycles = 0, extra_writeback_cycles = 0,
    host_issue_cycles = 0
  >
>} {}
)mlir";

TEST_F(TargetModelTest, BuildFromAttr) {
  auto result = buildFromText(kV4Config);
  ASSERT_TRUE(succeeded(result));
  const auto &model = *result;

  // Named memory spaces.
  ASSERT_EQ(model.memSpaces().size(), 2u);

  const auto *spad = model.findMemSpace("spad");
  ASSERT_NE(spad, nullptr);
  EXPECT_EQ(spad->capacityKib, 256);
  EXPECT_EQ(spad->capacityBytes(), 256 * 1024);
  EXPECT_EQ(spad->banks, 4);
  EXPECT_EQ(spad->rowWidthBytes, 16);
  // 256 KiB / (4 banks * 16 B/row) = 4096 rows per bank.
  EXPECT_EQ(spad->rowsPerBank(), 4096);

  const auto *acc = model.findMemSpace("acc");
  ASSERT_NE(acc, nullptr);
  EXPECT_EQ(acc->capacityKib, 64);
  EXPECT_EQ(acc->rowWidthBytes, 64);
  // 64 KiB / (4 banks * 64 B/row) = 256 rows per bank.
  EXPECT_EQ(acc->rowsPerBank(), 256);

  EXPECT_EQ(model.findMemSpace("nonexistent"), nullptr);

  // Executor instances (expanded from families).
  EXPECT_EQ(model.executors().size(), 3u);
  EXPECT_EQ(model.executors()[0].name(), "dma_read#0");
  EXPECT_EQ(model.executors()[1].name(), "dma_write#0");
  EXPECT_EQ(model.executors()[2].name(), "array#0");

  // Internal contention links.
  ASSERT_EQ(model.links().size(), 1u);
  EXPECT_EQ(model.links()[0].name(), "dram_bus#0");
}

TEST_F(TargetModelTest, PerFamilyConfig) {
  auto result = buildFromText(kV4Config);
  ASSERT_TRUE(succeeded(result));
  const auto &model = *result;

  // DMA families carry their own dma config.
  const auto *readDma = model.dmaConfig("dma_read");
  ASSERT_NE(readDma, nullptr);
  EXPECT_EQ(readDma->busBytesPerCycle, 16);
  EXPECT_EQ(readDma->fixedLatencyCycles, 20);
  EXPECT_EQ(readDma->maxOutstanding, 8);
  EXPECT_EQ(readDma->burstBytes, 64);

  const auto *writeDma = model.dmaConfig("dma_write");
  ASSERT_NE(writeDma, nullptr);

  // Wrong config type queries return nullptr.
  EXPECT_EQ(model.dmaConfig("array"), nullptr);
  EXPECT_EQ(model.arrayConfig("dma_read"), nullptr);
  EXPECT_EQ(model.dmaConfig("nonexistent"), nullptr);

  // Array family config.
  const auto *array = model.arrayConfig("array");
  ASSERT_NE(array, nullptr);
  EXPECT_EQ(array->topology, "systolic");
  EXPECT_EQ(array->meshRows, 16);
  EXPECT_EQ(array->meshCols, 16);
  EXPECT_TRUE(array->inputSkew);
  EXPECT_TRUE(array->interTilePipelined);
  EXPECT_EQ(array->numArrayInstances(), 1);
}

TEST_F(TargetModelTest, ExecutorConnectivity) {
  auto result = buildFromText(kV4Config);
  ASSERT_TRUE(succeeded(result));
  const auto &model = *result;

  EXPECT_TRUE(model.executorCanReach("array", "spad"));
  EXPECT_TRUE(model.executorCanReach("array", "acc"));
  EXPECT_FALSE(model.executorCanReach("array", "l2"));
  EXPECT_FALSE(model.executorCanReach("nonexistent", "spad"));

  // Links are internal: no connectivity, not findable as executors.
  const auto *dramBus = model.findLink("dram_bus", 0);
  ASSERT_NE(dramBus, nullptr);
  EXPECT_EQ(model.findExecutor("dram_bus", 0), nullptr);
}

TEST_F(TargetModelTest, DmaDerivedMethods) {
  DmaConfig dma{
      .busBytesPerCycle = 16,
      .fixedLatencyCycles = 20,
      .maxOutstanding = 8,
      .burstBytes = 64,
  };

  EXPECT_EQ(dma.transferCycles(512), 32);
  EXPECT_EQ(dma.transferCycles(1), 1);
  EXPECT_EQ(dma.transferCycles(16), 1);
  EXPECT_EQ(dma.transferCycles(17), 2);

  EXPECT_EQ(dma.totalLatency(512), 52);
  EXPECT_EQ(dma.totalLatency(0), 20);
}

TEST_F(TargetModelTest, FindExecutor) {
  const char *text = R"mlir(
module attributes {perf.target = #perf.target<
  version = 4,
  target_name = "test",
  frequency_mhz = 1000,
  memory_spaces = [
    #perf.memspace<
      name = "spad",
      capacity_kib = 256, banks = 4,
      read_ports = 4, write_ports = 4,
      port_width_bytes = 16, row_width_bytes = 16
    >
  ],
  executors = [
    #perf.executor_family<
      kind = "dma_read", count = 2, in_order = true, spaces = ["spad"],
      config = #perf.dma<bus_bytes_per_cycle = 16, fixed_latency_cycles = 20,
                         max_outstanding = 8, burst_bytes = 64>
    >,
    #perf.executor_family<
      kind = "array", count = 4, in_order = true, spaces = ["spad"],
      config = #perf.array<topology = "systolic",
                           mesh_rows = 16, mesh_cols = 16,
                           tile_rows = 2, tile_cols = 2,
                           pe_latency_cycles = 1,
                           input_skew = true,
                           inter_tile_pipelined = true>
    >
  ],
  links = [
  ],
  calibration = #perf.calibration<
    extra_fill_cycles = 0, extra_drain_cycles = 0,
    extra_preload_cycles = 0, extra_writeback_cycles = 0,
    host_issue_cycles = 0
  >
>} {}
)mlir";

  auto result = buildFromText(text);
  ASSERT_TRUE(succeeded(result));
  const auto &model = *result;

  EXPECT_EQ(model.executors().size(), 6u);
  EXPECT_EQ(model.executors()[0].name(), "dma_read#0");
  EXPECT_EQ(model.executors()[1].name(), "dma_read#1");
  EXPECT_EQ(model.executors()[2].name(), "array#0");
  EXPECT_EQ(model.executors()[5].name(), "array#3");

  auto *inst = model.findExecutor("dma_read", 1);
  ASSERT_NE(inst, nullptr);
  EXPECT_EQ(inst->name(), "dma_read#1");

  inst = model.findExecutor("array#3");
  ASSERT_NE(inst, nullptr);
  EXPECT_EQ(inst->kind, "array");
  EXPECT_EQ(inst->index, 3);

  // Array instances point back to their family's config.
  const auto *arrayFam = model.findFamily("array");
  ASSERT_NE(arrayFam, nullptr);
  EXPECT_EQ(arrayFam->arrayConfig->numArrayInstances(), 4);

  EXPECT_EQ(model.findExecutor("nonexistent", 0), nullptr);
  EXPECT_EQ(model.findExecutor("dma_read#99"), nullptr);
}

TEST_F(TargetModelTest, RejectsDuplicateSpaceNames) {
  const char *text = R"mlir(
module attributes {perf.target = #perf.target<
  version = 4,
  target_name = "test",
  frequency_mhz = 1000,
  memory_spaces = [
    #perf.memspace<
      name = "spad",
      capacity_kib = 256, banks = 4,
      read_ports = 4, write_ports = 4,
      port_width_bytes = 16, row_width_bytes = 16
    >,
    #perf.memspace<
      name = "spad",
      capacity_kib = 64, banks = 4,
      read_ports = 1, write_ports = 1,
      port_width_bytes = 64, row_width_bytes = 64
    >
  ],
  executors = [
    #perf.executor_family<
      kind = "array", count = 1, in_order = true, spaces = ["spad"],
      config = #perf.array<topology = "systolic",
                           mesh_rows = 16, mesh_cols = 16,
                           tile_rows = 1, tile_cols = 1,
                           pe_latency_cycles = 1,
                           input_skew = true,
                           inter_tile_pipelined = true>
    >
  ],
  links = [
  ],
  calibration = #perf.calibration<
    extra_fill_cycles = 0, extra_drain_cycles = 0,
    extra_preload_cycles = 0, extra_writeback_cycles = 0,
    host_issue_cycles = 0
  >
>} {}
)mlir";

  auto result = buildFromText(text);
  EXPECT_TRUE(failed(result));
}

TEST_F(TargetModelTest, RejectsUndeclaredSpaceReference) {
  const char *text = R"mlir(
module attributes {perf.target = #perf.target<
  version = 4,
  target_name = "test",
  frequency_mhz = 1000,
  memory_spaces = [
    #perf.memspace<
      name = "spad",
      capacity_kib = 256, banks = 4,
      read_ports = 4, write_ports = 4,
      port_width_bytes = 16, row_width_bytes = 16
    >
  ],
  executors = [
    #perf.executor_family<
      kind = "array", count = 1, in_order = true,
      spaces = ["spad", "typo_space"],
      config = #perf.array<topology = "systolic",
                           mesh_rows = 16, mesh_cols = 16,
                           tile_rows = 1, tile_cols = 1,
                           pe_latency_cycles = 1,
                           input_skew = true,
                           inter_tile_pipelined = true>
    >
  ],
  links = [
  ],
  calibration = #perf.calibration<
    extra_fill_cycles = 0, extra_drain_cycles = 0,
    extra_preload_cycles = 0, extra_writeback_cycles = 0,
    host_issue_cycles = 0
  >
>} {}
)mlir";

  auto result = buildFromText(text);
  EXPECT_TRUE(failed(result));
}

TEST_F(TargetModelTest, RejectsUnsupportedConfigType) {
  const char *text = R"mlir(
module attributes {perf.target = #perf.target<
  version = 4,
  target_name = "test",
  frequency_mhz = 1000,
  memory_spaces = [
    #perf.memspace<
      name = "spad",
      capacity_kib = 256, banks = 4,
      read_ports = 4, write_ports = 4,
      port_width_bytes = 16, row_width_bytes = 16
    >
  ],
  executors = [
    #perf.executor_family<
      kind = "array", count = 1, in_order = true, spaces = ["spad"],
      config = #perf.calibration<extra_fill_cycles = 0, extra_drain_cycles = 0,
                                 extra_preload_cycles = 0,
                                 extra_writeback_cycles = 0,
                                 host_issue_cycles = 0>
    >
  ],
  links = [
  ],
  calibration = #perf.calibration<
    extra_fill_cycles = 0, extra_drain_cycles = 0,
    extra_preload_cycles = 0, extra_writeback_cycles = 0,
    host_issue_cycles = 0
  >
>} {}
)mlir";

  auto result = buildFromText(text);
  EXPECT_TRUE(failed(result));
}

TEST_F(TargetModelTest, RejectsDuplicateExecutorKinds) {
  const char *text = R"mlir(
module attributes {perf.target = #perf.target<
  version = 4,
  target_name = "test",
  frequency_mhz = 1000,
  memory_spaces = [
    #perf.memspace<
      name = "spad",
      capacity_kib = 256, banks = 4,
      read_ports = 4, write_ports = 4,
      port_width_bytes = 16, row_width_bytes = 16
    >
  ],
  executors = [
    #perf.executor_family<
      kind = "array", count = 1, in_order = true, spaces = ["spad"],
      config = #perf.array<topology = "systolic",
                           mesh_rows = 16, mesh_cols = 16,
                           tile_rows = 1, tile_cols = 1,
                           pe_latency_cycles = 1,
                           input_skew = true,
                           inter_tile_pipelined = true>
    >,
    #perf.executor_family<
      kind = "array", count = 1, in_order = true, spaces = ["spad"],
      config = #perf.array<topology = "systolic",
                           mesh_rows = 8, mesh_cols = 8,
                           tile_rows = 1, tile_cols = 1,
                           pe_latency_cycles = 1,
                           input_skew = true,
                           inter_tile_pipelined = true>
    >
  ],
  links = [
  ],
  calibration = #perf.calibration<
    extra_fill_cycles = 0, extra_drain_cycles = 0,
    extra_preload_cycles = 0, extra_writeback_cycles = 0,
    host_issue_cycles = 0
  >
>} {}
)mlir";

  auto result = buildFromText(text);
  EXPECT_TRUE(failed(result));
}
