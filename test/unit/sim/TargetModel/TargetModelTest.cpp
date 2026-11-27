// TargetModelTest.cpp - Unit tests for TargetModel.
//
// Two test styles, by design:
//
//   - Dump-driven: the model's dump() is its full serialized surface, so
//     comparing dumps against a checked-in expected file (or against the
//     model built from an equivalent config) covers every field at once.
//     The per-field API assertions that would duplicate this are omitted.
//   - Explicit API tests: kept only where the dump cannot observe the
//     behavior — derived computations, lookup-miss semantics, build-time
//     rejection, and the R1 reachability predicate.
//
// Fixtures live under data/<feature>/ (target.mlir, optional config.json,
// optional expected.dump); the shared prior is data/prior.mlir.

#include "TestUtils.hpp"

using namespace mlir;
using namespace perf;
using namespace perftest;

using TargetModelTest = TargetModelTestBase;

//===----------------------------------------------------------------------===//
// Dump-driven: whole-model surface
//===----------------------------------------------------------------------===//

// The baseline model's dump matches the checked-in expectation. This one
// comparison covers memory-space geometry, link bandwidth, executor
// expansion/naming, and both per-family configs.
TEST_F(TargetModelTest, BaselineDump) {
  auto model = buildFromFeatureTarget("baseline");
  ASSERT_TRUE(succeeded(model));
  EXPECT_EQ(dumpToString(*model),
            readFile(featureDir("baseline") + "/expected.dump"));
}

// The architect-level config.json and the simulator-level target.mlir it
// lowers to must build into the same TargetModel.
TEST_F(TargetModelTest, JsonMatchesMlir) {
  auto jsonModel = buildFromFeatureConfig("baseline");
  ASSERT_TRUE(succeeded(jsonModel));
  auto mlirModel = buildFromFeatureTarget("baseline");
  ASSERT_TRUE(succeeded(mlirModel));
  EXPECT_EQ(dumpToString(*jsonModel), dumpToString(*mlirModel));
}

// Multi-instance expansion: array count = 2, dma count = 2, two DRAM
// channels. Per-instance naming and ordering are visible in the dump.
TEST_F(TargetModelTest, MultiArrayDump) {
  auto model = buildFromFeatureTarget("multi_array");
  ASSERT_TRUE(succeeded(model));
  EXPECT_EQ(dumpToString(*model),
            readFile(featureDir("multi_array") + "/expected.dump"));
}

//===----------------------------------------------------------------------===//
// Explicit API tests: behavior the dump does not observe
//===----------------------------------------------------------------------===//

// Derived cycle formulas on DmaConfig (transfer bandwidth comes from the
// dram link, not the engine).
TEST_F(TargetModelTest, DmaDerivedMethods) {
  DmaConfig dma{
      .readPorts = 1,
      .writePorts = 1,
      .fixedLatencyCycles = 20,
      .maxOutstanding = 8,
      .burstBytes = 64,
  };
  const int64_t bw = 16;

  EXPECT_EQ(dma.transferCycles(512, bw), 32);
  EXPECT_EQ(dma.transferCycles(1, bw), 1);
  EXPECT_EQ(dma.transferCycles(16, bw), 1);
  EXPECT_EQ(dma.transferCycles(17, bw), 2);

  EXPECT_EQ(dma.totalLatency(512, bw), 52);
  EXPECT_EQ(dma.totalLatency(0, bw), 20);
}

// Lookup misses and wrong-kind queries return an empty OptCRef.
TEST_F(TargetModelTest, LookupMisses) {
  auto model = buildFromFeatureTarget("baseline");
  ASSERT_TRUE(succeeded(model));

  EXPECT_FALSE(model->findMemSpace("nonexistent").has_value());
  EXPECT_FALSE(model->findExecutor("nonexistent", 0).has_value());
  EXPECT_FALSE(model->findExecutor("dma_read#99").has_value());
  // Links are internal: not findable as executors.
  EXPECT_FALSE(model->findExecutor("dram_bus", 0).has_value());
  // Wrong config type queries return empty.
  EXPECT_FALSE(model->dmaConfig("array").has_value());
  EXPECT_FALSE(model->arrayConfig("dma_read").has_value());
  EXPECT_FALSE(model->dmaConfig("nonexistent").has_value());

  // Hit-by-name lookup agrees with hit-by-kind/index (same instance).
  auto byName = model->findExecutor("array#0");
  auto byKind = model->findExecutor("array", 0);
  ASSERT_TRUE(byName.has_value());
  ASSERT_TRUE(byKind.has_value());
  EXPECT_EQ(&*byName, &*byKind);
}

// R1 connectivity predicate.
TEST_F(TargetModelTest, ExecutorConnectivity) {
  auto model = buildFromFeatureTarget("baseline");
  ASSERT_TRUE(succeeded(model));

  EXPECT_TRUE(model->executorCanReach("array", "spad"));
  EXPECT_TRUE(model->executorCanReach("array", "acc"));
  EXPECT_FALSE(model->executorCanReach("array", "l2"));
  EXPECT_FALSE(model->executorCanReach("nonexistent", "spad"));
}

// Duplicate memory-space names are rejected at build time.
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
  executors = [],
  links = []
>} {}
)mlir";

  EXPECT_TRUE(failed(buildFromText(text)));
}
