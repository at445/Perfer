// ExecutionViewTest.cpp - Unit tests for ExecutionView.
//
// Programs under test are fixtures: data/<feature>/input/program.mlir.
// The view has no serialized form, so assertions are on the index it
// builds from those programs.

#include "TestUtils.hpp"

#include "perf/Sim/ExecutionView.h"
#include "perf/Sim/DependencyGraph.h"

using namespace mlir;
using namespace perf;
using namespace perftest;

namespace {

// The token_chain program and its ExecutionView, shared by the indexing
// tests.
class ExecutionViewTest : public SimViewTestBase {
protected:
  void SetUp() override {
    SimViewTestBase::SetUp();
    module = parseProgram("token_chain");
    ASSERT_TRUE(module);
    view = ExecutionView::build(*module);
  }

  OwningOpRef<ModuleOp> module;
  ExecutionView view;
};

} // namespace

TEST_F(ExecutionViewTest, IndexesAllExecutorOpsInProgramOrder) {
  ASSERT_EQ(view.ops().size(), 5u); // la, lb, i0, c0, st

  // Ops are in block program order; ids are dense and stable.
  for (OpId id = 0; id < view.ops().size(); ++id) {
    Operation *op = view.opOf(id);
    EXPECT_EQ(view.idOf(op), id);
  }
}

TEST_F(ExecutionViewTest, PlacementParsedFromResourceAttr) {
  EXPECT_EQ(view.placementOf(view.opOf(0)), ResourceId("dma_read", 0));
  EXPECT_EQ(view.placementOf(view.opOf(1)), ResourceId("dma_read", 0));
  EXPECT_EQ(view.placementOf(view.opOf(2)), ResourceId("array", 0));
  EXPECT_EQ(view.placementOf(view.opOf(3)), ResourceId("array", 0));
  EXPECT_EQ(view.placementOf(view.opOf(4)), ResourceId("dma_write", 0));
}

TEST_F(ExecutionViewTest, TokenProducerCoversExecutorResults) {
  const auto &prod = view.tokenProducer();
  // Each executor op's result token maps back to itself.
  for (Operation *op : view.ops())
    EXPECT_EQ(prod.at(op->getResult(0)), op);
  // token.init is not a producer.
  EXPECT_EQ(prod.size(), 5u);
}

// Provenance-only ops (perf.alloc, memref.subview, token.init, fence) are
// transparent to the view: they get no OpId, no placement, and do not
// become DependencyGraph nodes even when an executor op reads through them.
TEST_F(SimViewTestBase, ProvenanceOnlyOpsAreNotIndexed) {
  auto mod = parseProgram("provenance_ops");
  ASSERT_TRUE(mod);
  auto view = ExecutionView::build(*mod);

  // Only the four executor ops are indexed, in program order.
  ASSERT_EQ(view.ops().size(), 4u); // la, i0, c0, st
  EXPECT_TRUE(isa<DmaLoadOp>(view.opOf(0)));
  EXPECT_TRUE(isa<ArrayInitOp>(view.opOf(1)));
  EXPECT_TRUE(isa<ComputeOp>(view.opOf(2)));
  EXPECT_TRUE(isa<DmaStoreOp>(view.opOf(3)));

  // Non-executor ops resolve to the invalid id.
  mod->walk([&](Operation *op) {
    if (isa<memref::SubViewOp>(op)) {
      EXPECT_EQ(view.idOf(op), ExecutionView::kInvalidOpId);
    }
  });

  // The compute reads through subviews but depends only on la and i0.
  auto deps = DependencyGraph::build(view);
  EXPECT_EQ(deps.predecessors(2).size(), 2u); // c0 <- {la, i0}
  EXPECT_EQ(deps.predecessors(3).size(), 1u); // st <- c0
}
