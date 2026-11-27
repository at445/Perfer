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

  PriorAttr getPriorAttr(ModuleOp mod) {
    return llvm::dyn_cast_if_present<PriorAttr>(
        mod->getAttr("perf.prior"));
  }

  FailureOr<TargetModel> buildFromText(StringRef targetText,
                                       StringRef priorText) {
    auto targetMod = parseModule(targetText);
    if (!targetMod)
      return failure();
    auto target = getTargetAttr(targetMod);
    if (!target)
      return failure();

    auto priorMod = parseModule(priorText);
    if (!priorMod)
      return failure();
    auto prior = getPriorAttr(priorMod);
    if (!prior)
      return failure();

    return TargetModel::build(target, prior);
  }
};

// Shared prior used by all tests.
static constexpr const char *kPrior = R"mlir(
module attributes {perf.prior = #perf.prior<
  version = 4,
  target_name = "test_prior",
  calibration = #perf.calibration<
    extra_fill_cycles = 0, extra_drain_cycles = 0,
    extra_preload_cycles = 0, extra_writeback_cycles = 0,
    host_issue_cycles = 0
  >
>} {}
)mlir";

// Flat #perf.target baseline config: per-family configs carried inline.
static constexpr const char *kV4Config = R"mlir(
module attributes {perf.target = #perf.target<
  version = 4,
  target_name = "test_npu",
  frequency_mhz = 1000,
  memory_spaces = [
    #perf.sram<
      name = "spad",
      capacity_kib = 256, banks = 4,
      read_ports = 4, write_ports = 4,
      port_width_bytes = 16, row_width_bytes = 16, interleave = "low"
    >,
    #perf.sram<
      name = "acc",
      capacity_kib = 64, banks = 4,
      read_ports = 1, write_ports = 1,
      port_width_bytes = 64, row_width_bytes = 64, interleave = "low"
    >
  ],
  executors = [
    #perf.executor<
      kind = "dma_read", count = 1,
      spaces = ["spad", "acc"],
      config = #perf.dma<read_ports = 1, write_ports = 1,
                         fixed_latency_cycles = 20,
                         max_outstanding = 8, burst_bytes = 64>
    >,
    #perf.executor<
      kind = "dma_write", count = 1,
      spaces = ["spad", "acc"],
      config = #perf.dma<read_ports = 1, write_ports = 1,
                         fixed_latency_cycles = 20,
                         max_outstanding = 8, burst_bytes = 64>
    >,
    #perf.executor<
      kind = "array", count = 1,
      spaces = ["spad", "acc"],
      config = #perf.array<topology = "systolic",
                           mesh_rows = 16, mesh_cols = 16,
                           tile_rows = 1, tile_cols = 1,
                           pe_latency_cycles = 1,
                           input_skew = true,
                           inter_tile_pipelined = true,
                           acc_read_scale_cycles = 2,
                           acc_read_act_cycles = 1>
    >
  ],
  links = [
    #perf.link<kind = "dram_bus", count = 1, bytes_per_cycle = 16>
  ]
>} {}
)mlir";

TEST_F(TargetModelTest, BuildFromAttr) {
  auto result = buildFromText(kV4Config, kPrior);
  ASSERT_TRUE(succeeded(result));
  const auto &model = *result;

  // Named memory spaces.
  ASSERT_EQ(model.memSpaces().size(), 2u);

  const auto spad = model.findMemSpace("spad");
  ASSERT_TRUE(spad.has_value());
  EXPECT_EQ(spad->capacityKib, 256);
  EXPECT_EQ(spad->capacityBytes(), 256 * 1024);
  EXPECT_EQ(spad->banks, 4);
  EXPECT_EQ(spad->rowWidthBytes, 16);
  // 256 KiB / (4 banks * 16 B/row) = 4096 rows per bank.
  EXPECT_EQ(spad->rowsPerBank(), 4096);

  const auto acc = model.findMemSpace("acc");
  ASSERT_TRUE(acc.has_value());
  EXPECT_EQ(acc->capacityKib, 64);
  EXPECT_EQ(acc->rowWidthBytes, 64);
  // 64 KiB / (4 banks * 64 B/row) = 256 rows per bank.
  EXPECT_EQ(acc->rowsPerBank(), 256);

  EXPECT_FALSE(model.findMemSpace("nonexistent").has_value());

  // Executor instances (expanded from the executor entries).
  EXPECT_EQ(model.executors().size(), 3u);
  EXPECT_EQ(model.executors()[0].name(), "dma_read#0");
  EXPECT_EQ(model.executors()[1].name(), "dma_write#0");
  EXPECT_EQ(model.executors()[2].name(), "array#0");

  // Internal contention links.
  ASSERT_EQ(model.links().size(), 1u);
  EXPECT_EQ(model.links()[0].name(), "dram_bus#0");
}

TEST_F(TargetModelTest, PerFamilyConfig) {
  auto result = buildFromText(kV4Config, kPrior);
  ASSERT_TRUE(succeeded(result));
  const auto &model = *result;

  // DMA instances carry their engine-intrinsic config (bandwidth lives on
  // the dram_bus link, not the engine).
  const auto readDma = model.dmaConfig("dma_read");
  ASSERT_TRUE(readDma.has_value());
  EXPECT_EQ(readDma->fixedLatencyCycles, 20);
  EXPECT_EQ(readDma->maxOutstanding, 8);
  EXPECT_EQ(readDma->burstBytes, 64);

  const auto writeDma = model.dmaConfig("dma_write");
  ASSERT_TRUE(writeDma.has_value());

  // Wrong config type queries return std::nullopt.
  EXPECT_FALSE(model.dmaConfig("array").has_value());
  EXPECT_FALSE(model.arrayConfig("dma_read").has_value());
  EXPECT_FALSE(model.dmaConfig("nonexistent").has_value());

  // Array config, including the acc read-out epilogue fields.
  const auto array = model.arrayConfig("array");
  ASSERT_TRUE(array.has_value());
  EXPECT_EQ(array->topology, "systolic");
  EXPECT_EQ(array->meshRows, 16);
  EXPECT_EQ(array->meshCols, 16);
  EXPECT_TRUE(array->inputSkew);
  EXPECT_TRUE(array->interTilePipelined);
  EXPECT_EQ(array->accReadScaleCycles, 2);
  EXPECT_EQ(array->accReadActCycles, 1);
  EXPECT_EQ(array->numArrayInstances(), 1);

  // DRAM channel bandwidth is looked up from the dram_bus link.
  EXPECT_EQ(model.dramBytesPerCycle(), 16);
}

TEST_F(TargetModelTest, ExecutorConnectivity) {
  auto result = buildFromText(kV4Config, kPrior);
  ASSERT_TRUE(succeeded(result));
  const auto &model = *result;

  EXPECT_TRUE(model.executorCanReach("array", "spad"));
  EXPECT_TRUE(model.executorCanReach("array", "acc"));
  EXPECT_FALSE(model.executorCanReach("array", "l2"));
  EXPECT_FALSE(model.executorCanReach("nonexistent", "spad"));

  // Links are internal: no connectivity, not findable as executors.
  const auto dramBus = model.findLink("dram_bus", 0);
  ASSERT_TRUE(dramBus.has_value());
  EXPECT_FALSE(model.findExecutor("dram_bus", 0).has_value());
}

TEST_F(TargetModelTest, DmaDerivedMethods) {
  DmaConfig dma{
      .readPorts = 1,
      .writePorts = 1,
      .fixedLatencyCycles = 20,
      .maxOutstanding = 8,
      .burstBytes = 64,
  };
  // Transfer bandwidth comes from the dram link the engine's ports share.
  const int64_t bw = 16;

  EXPECT_EQ(dma.transferCycles(512, bw), 32);
  EXPECT_EQ(dma.transferCycles(1, bw), 1);
  EXPECT_EQ(dma.transferCycles(16, bw), 1);
  EXPECT_EQ(dma.transferCycles(17, bw), 2);

  EXPECT_EQ(dma.totalLatency(512, bw), 52);
  EXPECT_EQ(dma.totalLatency(0, bw), 20);
}

TEST_F(TargetModelTest, FindExecutor) {
  const char *text = R"mlir(
module attributes {perf.target = #perf.target<
  version = 4,
  target_name = "test",
  frequency_mhz = 1000,
  memory_spaces = [
    #perf.sram<
      name = "spad",
      capacity_kib = 256, banks = 4,
      read_ports = 4, write_ports = 4,
      port_width_bytes = 16, row_width_bytes = 16, interleave = "low"
    >
  ],
  executors = [
    #perf.executor<
      kind = "dma_read", count = 2,
      spaces = ["spad"],
      config = #perf.dma<read_ports = 2, write_ports = 0,
                         fixed_latency_cycles = 20,
                         max_outstanding = 8, burst_bytes = 64>
    >,
    #perf.executor<
      kind = "array", count = 4,
      spaces = ["spad"],
      config = #perf.array<topology = "systolic",
                           mesh_rows = 16, mesh_cols = 16,
                           tile_rows = 2, tile_cols = 2,
                           pe_latency_cycles = 1,
                           input_skew = true,
                           inter_tile_pipelined = true,
                           acc_read_scale_cycles = 0,
                           acc_read_act_cycles = 0>
    >
  ],
  links = []
>} {}
)mlir";

  auto result = buildFromText(text, kPrior);
  ASSERT_TRUE(succeeded(result));
  const auto &model = *result;

  EXPECT_EQ(model.executors().size(), 6u);
  EXPECT_EQ(model.executors()[0].name(), "dma_read#0");
  EXPECT_EQ(model.executors()[1].name(), "dma_read#1");
  EXPECT_EQ(model.executors()[2].name(), "array#0");
  EXPECT_EQ(model.executors()[5].name(), "array#3");

  auto inst = model.findExecutor("dma_read", 1);
  ASSERT_TRUE(inst.has_value());
  EXPECT_EQ(inst->name(), "dma_read#1");

  inst = model.findExecutor("array#3");
  ASSERT_TRUE(inst.has_value());
  EXPECT_EQ(inst->kind, "array");
  EXPECT_EQ(inst->index, 3);

  // Array instances carry their own array config.
  const auto arrayCfg = model.arrayConfig("array");
  ASSERT_TRUE(arrayCfg.has_value());
  EXPECT_EQ(arrayCfg->numArrayInstances(), 4);

  EXPECT_FALSE(model.findExecutor("nonexistent", 0).has_value());
  EXPECT_FALSE(model.findExecutor("dma_read#99").has_value());
}

TEST_F(TargetModelTest, RejectsDuplicateSpaceNames) {
  const char *text = R"mlir(
module attributes {perf.target = #perf.target<
  version = 4,
  target_name = "test",
  frequency_mhz = 1000,
  memory_spaces = [
    #perf.sram<
      name = "spad",
      capacity_kib = 256, banks = 4,
      read_ports = 4, write_ports = 4,
      port_width_bytes = 16, row_width_bytes = 16, interleave = "low"
    >,
    #perf.sram<
      name = "spad",
      capacity_kib = 64, banks = 4,
      read_ports = 1, write_ports = 1,
      port_width_bytes = 64, row_width_bytes = 64, interleave = "low"
    >
  ],
  executors = [
    #perf.executor<
      kind = "array", count = 1,
      spaces = ["spad"],
      config = #perf.array<topology = "systolic",
                           mesh_rows = 16, mesh_cols = 16,
                           tile_rows = 1, tile_cols = 1,
                           pe_latency_cycles = 1,
                           input_skew = true,
                           inter_tile_pipelined = true,
                           acc_read_scale_cycles = 0,
                           acc_read_act_cycles = 0>
    >
  ],
  links = []
>} {}
)mlir";

  auto result = buildFromText(text, kPrior);
  EXPECT_TRUE(failed(result));
}
