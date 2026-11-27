// ExecutionViewTest.cpp - Unit tests for the simulator's op-view layer
// (ExecutionView / DependencyGraph / ScheduleView).
//
// Programs under test are fixtures: data/<feature>/input/program.mlir.
// The views have no serialized form (yet), so assertions are on the C++
// structures they build from those programs.

#include "perf/Sim/ExecutionView.h"
#include "perf/Sim/DependencyGraph.h"
#include "perf/Sim/ScheduleView.h"

#include "perf/Dialect/PerfDialect.h"
#include "perf/Dialect/PerfOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "llvm/Support/MemoryBuffer.h"

#include "gtest/gtest.h"

using namespace mlir;
using namespace perf;

namespace {

// Fixture root: data/<feature>/input/program.mlir.
const std::string kDataDir = PERF_UNIT_TEST_DATA_DIR;

std::string programPath(llvm::StringRef feature) {
  return kDataDir + "/" + feature.str() + "/input/program.mlir";
}

class SimViewTest : public ::testing::Test {
protected:
  void SetUp() override {
    context.loadDialect<PerfDialect>();
    context.loadDialect<func::FuncDialect>();
    context.loadDialect<memref::MemRefDialect>();
  }

  // Parse a feature's input program; null ref when unreadable/invalid.
  OwningOpRef<ModuleOp> parseProgram(llvm::StringRef feature) {
    auto bufferOr = llvm::MemoryBuffer::getFile(programPath(feature));
    if (!bufferOr)
      return nullptr;
    return parseSourceString<ModuleOp>((*bufferOr)->getBuffer(), &context);
  }

  MLIRContext context;
};

// The default fixture: the token_chain program and its ExecutionView,
// shared by the ExecutionView / DependencyGraph / default ScheduleView
// tests.
class TokenChainTest : public SimViewTest {
protected:
  void SetUp() override {
    SimViewTest::SetUp();
    module = parseProgram("token_chain");
    ASSERT_TRUE(module);
    view = ExecutionView::build(*module);
  }

  OwningOpRef<ModuleOp> module;
  ExecutionView view;
};

} // namespace

//===----------------------------------------------------------------------===//
// ExecutionView
//===----------------------------------------------------------------------===//

TEST_F(TokenChainTest, IndexesAllExecutorOpsInProgramOrder) {
  ASSERT_EQ(view.ops().size(), 5u); // la, lb, i0, c0, st

  // Ops are in block program order; ids are dense and stable.
  for (OpId id = 0; id < view.ops().size(); ++id) {
    Operation *op = view.opOf(id);
    EXPECT_EQ(view.idOf(op), id);
  }
}

TEST_F(TokenChainTest, PlacementParsedFromResourceAttr) {
  EXPECT_EQ(view.placementOf(view.opOf(0)), ResourceId("dma_read", 0));
  EXPECT_EQ(view.placementOf(view.opOf(1)), ResourceId("dma_read", 0));
  EXPECT_EQ(view.placementOf(view.opOf(2)), ResourceId("array", 0));
  EXPECT_EQ(view.placementOf(view.opOf(3)), ResourceId("array", 0));
  EXPECT_EQ(view.placementOf(view.opOf(4)), ResourceId("dma_write", 0));
}

TEST_F(TokenChainTest, TokenProducerCoversExecutorResults) {
  const auto &prod = view.tokenProducer();
  // Each executor op's result token maps back to itself.
  for (Operation *op : view.ops())
    EXPECT_EQ(prod.at(op->getResult(0)), op);
  // token.init is not a producer.
  EXPECT_EQ(prod.size(), 5u);
}

//===----------------------------------------------------------------------===//
// DependencyGraph
//===----------------------------------------------------------------------===//

TEST_F(TokenChainTest, TokenEdgesFormExpectedGraph) {
  auto deps = DependencyGraph::build(view);

  // la(0) and lb(1) and i0(2) are sources; c0(3) depends on all three;
  // st(4) depends on c0.
  EXPECT_TRUE(deps.predecessors(0).empty());
  EXPECT_TRUE(deps.predecessors(1).empty());
  EXPECT_TRUE(deps.predecessors(2).empty());

  ASSERT_EQ(deps.predecessors(3).size(), 3u);
  EXPECT_EQ(deps.predecessors(4).size(), 1u);
  EXPECT_EQ(deps.predecessors(4)[0], 3u);

  EXPECT_EQ(deps.successors(0).size(), 1u); // la -> c0
  EXPECT_EQ(deps.successors(3).size(), 1u); // c0 -> st
}

TEST_F(TokenChainTest, ReadyTracksDoneMarks) {
  auto deps = DependencyGraph::build(view);

  EXPECT_TRUE(deps.readyByDeps(0));
  EXPECT_TRUE(deps.readyByDeps(1));
  EXPECT_TRUE(deps.readyByDeps(2));
  EXPECT_FALSE(deps.readyByDeps(3)); // needs la, lb, i0
  EXPECT_FALSE(deps.readyByDeps(4)); // needs c0

  deps.markDone(0, 36);
  deps.markDone(1, 68);
  EXPECT_FALSE(deps.readyByDeps(3)); // i0 still undone
  deps.markDone(2, 16);
  EXPECT_TRUE(deps.readyByDeps(3));
  EXPECT_FALSE(deps.readyByDeps(4));

  deps.markDone(3, 100);
  EXPECT_TRUE(deps.readyByDeps(4));
  EXPECT_EQ(deps.doneTime(3), 100u);
}

//===----------------------------------------------------------------------===//
// ScheduleView
//===----------------------------------------------------------------------===//

TEST_F(TokenChainTest, DefaultOrderIsProgramOrderPerResource) {
  auto sched = ScheduleView::build(view);

  // dma_read#0: la(0) then lb(1) (program order).
  ResourceId dmaRead{"dma_read", 0};
  EXPECT_EQ(sched.head(dmaRead), 0u);
  sched.advance(dmaRead);
  EXPECT_EQ(sched.head(dmaRead), 1u);
  sched.advance(dmaRead);
  EXPECT_TRUE(sched.isExhausted(dmaRead));

  // array#0: i0(2) then c0(3).
  ResourceId array{"array", 0};
  EXPECT_EQ(sched.head(array), 2u);
  sched.advance(array);
  EXPECT_EQ(sched.head(array), 3u);

  // dma_write#0: st(4) only.
  ResourceId dmaWrite{"dma_write", 0};
  EXPECT_EQ(sched.head(dmaWrite), 4u);
  sched.advance(dmaWrite);
  EXPECT_TRUE(sched.isExhausted(dmaWrite));

  // Unknown resource is exhausted.
  EXPECT_TRUE(sched.isExhausted({"nope", 0}));
}

// Explicit order annotations invert program order; uses its own fixture.
TEST_F(SimViewTest, ExplicitOrderAnnotationOverridesProgramOrder) {
  auto mod = parseProgram("explicit_order");
  ASSERT_TRUE(mod);
  auto view = ExecutionView::build(*mod);
  auto sched = ScheduleView::build(view);

  // Both ops annotated: order=1 (lb, id 1) precedes order=2 (la, id 0).
  ResourceId dmaRead{"dma_read", 0};
  EXPECT_EQ(sched.head(dmaRead), 1u);
  sched.advance(dmaRead);
  EXPECT_EQ(sched.head(dmaRead), 0u);
}
