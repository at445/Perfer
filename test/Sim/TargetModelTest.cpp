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
  void SetUp() override {
    context.loadDialect<PerfDialect>();
  }

  MLIRContext context;

  ModuleOp parseModule(StringRef text) {
    return parseSourceString<ModuleOp>(text, &context).get();
  }

  TargetAttr getTargetAttr(ModuleOp mod) {
    return llvm::dyn_cast_if_present<TargetAttr>(
        mod->getAttr("perf.target"));
  }
};

TEST_F(TargetModelTest, BuildFromAttr) {
  const char *text = R"mlir(
module attributes {perf.target = #perf.target<
  version = 3,
  target_name = "test_npu",
  frequency_mhz = 1000,
  array = #perf.array<
    topology = "systolic",
    mesh_rows = 16, mesh_cols = 16,
    tile_rows = 1, tile_cols = 1,
    pe_latency_cycles = 1,
    input_skew = true,
    inter_tile_pipelined = true
  >,
  scratchpad = #perf.sram<
    capacity_kib = 256, banks = 4,
    read_ports = 4, write_ports = 4, port_width_bytes = 16
  >,
  accumulator = #perf.sram<
    capacity_kib = 64, banks = 4,
    read_ports = 1, write_ports = 1, port_width_bytes = 64
  >,
  dma = #perf.dma<
    bus_bytes_per_cycle = 16, fixed_latency_cycles = 20,
    max_outstanding = 8, burst_bytes = 64
  >,
  queues = #perf.queues<
    load_queue_depth = 8, store_queue_depth = 8,
    execute_queue_depth = 8, dependency_entries = 32
  >,
  resources = [
    #perf.resource_family<kind = "dma_read",  count = 1, concurrency = 1, in_order = true, placeable = true>,
    #perf.resource_family<kind = "dma_write", count = 1, concurrency = 1, in_order = true, placeable = true>,
    #perf.resource_family<kind = "array",     count = 1, concurrency = 1, in_order = true, placeable = true>,
    #perf.resource_family<kind = "dram_bus",  count = 1, concurrency = 1, in_order = true, placeable = false>
  ],
  calibration = #perf.calibration<
    extra_fill_cycles = 0, extra_drain_cycles = 0,
    extra_preload_cycles = 0, extra_writeback_cycles = 0,
    host_issue_cycles = 0
  >
>} {}
)mlir";

  auto mod = parseModule(text);
  ASSERT_TRUE(mod);

  auto attr = getTargetAttr(mod);
  ASSERT_TRUE(attr);

  auto result = TargetModel::build(attr);
  ASSERT_TRUE(succeeded(result));
  const auto &model = *result;

  EXPECT_EQ(model.dma().busBytesPerCycle, 16);
  EXPECT_EQ(model.dma().fixedLatencyCycles, 20);
  EXPECT_EQ(model.dma().maxOutstanding, 8);
  EXPECT_EQ(model.dma().burstBytes, 64);

  EXPECT_EQ(model.array().topology, "systolic");
  EXPECT_EQ(model.array().meshRows, 16);
  EXPECT_EQ(model.array().meshCols, 16);
  EXPECT_EQ(model.array().tileRows, 1);
  EXPECT_EQ(model.array().tileCols, 1);
  EXPECT_TRUE(model.array().inputSkew);
  EXPECT_TRUE(model.array().interTilePipelined);
  EXPECT_EQ(model.array().numArrayInstances(), 1);

  EXPECT_EQ(model.scratchpad().capacityKib, 256);
  EXPECT_EQ(model.scratchpad().capacityBytes(), 256 * 1024);

  EXPECT_EQ(model.accumulator().capacityKib, 64);
  EXPECT_EQ(model.accumulator().capacityBytes(), 64 * 1024);

  EXPECT_EQ(model.instances().size(), 4u);
  EXPECT_EQ(model.instances()[0].name(), "dma_read#0");
  EXPECT_EQ(model.instances()[1].name(), "dma_write#0");
  EXPECT_EQ(model.instances()[2].name(), "array#0");
  EXPECT_EQ(model.instances()[3].name(), "dram_bus#0");

  EXPECT_TRUE(model.instances()[0].placeable);
  EXPECT_TRUE(model.instances()[3].placeable == false);
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
  EXPECT_EQ(dma.transferCycles(15), 1);
  EXPECT_EQ(dma.transferCycles(16), 1);
  EXPECT_EQ(dma.transferCycles(17), 2);

  EXPECT_EQ(dma.totalLatency(512), 52);
  EXPECT_EQ(dma.totalLatency(0), 20);
}

TEST_F(TargetModelTest, FindInstance) {
  const char *text = R"mlir(
module attributes {perf.target = #perf.target<
  version = 3,
  target_name = "test",
  frequency_mhz = 1000,
  array = #perf.array<
    topology = "systolic",
    mesh_rows = 16, mesh_cols = 16,
    tile_rows = 2, tile_cols = 2,
    pe_latency_cycles = 1,
    input_skew = true,
    inter_tile_pipelined = true
  >,
  scratchpad = #perf.sram<
    capacity_kib = 256, banks = 4,
    read_ports = 4, write_ports = 4, port_width_bytes = 16
  >,
  accumulator = #perf.sram<
    capacity_kib = 64, banks = 4,
    read_ports = 1, write_ports = 1, port_width_bytes = 64
  >,
  dma = #perf.dma<
    bus_bytes_per_cycle = 16, fixed_latency_cycles = 20,
    max_outstanding = 8, burst_bytes = 64
  >,
  queues = #perf.queues<
    load_queue_depth = 8, store_queue_depth = 8,
    execute_queue_depth = 8, dependency_entries = 32
  >,
  resources = [
    #perf.resource_family<kind = "dma_read",  count = 2, concurrency = 1, in_order = true, placeable = true>,
    #perf.resource_family<kind = "array",     count = 4, concurrency = 1, in_order = true, placeable = true>
  ],
  calibration = #perf.calibration<
    extra_fill_cycles = 0, extra_drain_cycles = 0,
    extra_preload_cycles = 0, extra_writeback_cycles = 0,
    host_issue_cycles = 0
  >
>} {}
)mlir";

  auto mod = parseModule(text);
  ASSERT_TRUE(mod);

  auto attr = getTargetAttr(mod);
  ASSERT_TRUE(attr);

  auto result = TargetModel::build(attr);
  ASSERT_TRUE(succeeded(result));
  const auto &model = *result;

  EXPECT_EQ(model.instances().size(), 6u);
  EXPECT_EQ(model.instances()[0].name(), "dma_read#0");
  EXPECT_EQ(model.instances()[1].name(), "dma_read#1");
  EXPECT_EQ(model.instances()[2].name(), "array#0");
  EXPECT_EQ(model.instances()[5].name(), "array#3");

  auto *inst = model.findInstance("dma_read", 1);
  ASSERT_NE(inst, nullptr);
  EXPECT_EQ(inst->name(), "dma_read#1");

  inst = model.findInstance("array#3");
  ASSERT_NE(inst, nullptr);
  EXPECT_EQ(inst->kind, "array");
  EXPECT_EQ(inst->index, 3);

  EXPECT_EQ(model.findInstance("nonexistent", 0), nullptr);
  EXPECT_EQ(model.findInstance("dma_read#99"), nullptr);
}
