// DependencyGraphTest.cpp - Unit tests for DependencyGraph.
//
// The token dependencies of the token_chain program.

#include "TestUtils.hpp"

#include "perf/Sim/ExecutionView.h"
#include "perf/Sim/DependencyGraph.h"

using namespace mlir;
using namespace perf;
using namespace perftest;

namespace {

class DependencyGraphTest : public SimViewTestBase {
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

TEST_F(DependencyGraphTest, TokenEdgesFormExpectedGraph) {
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

TEST_F(DependencyGraphTest, ReadyTracksDoneMarks) {
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
