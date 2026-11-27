// ScheduleViewTest.cpp - Unit tests for ScheduleView.
//
// Default per-resource order from token_chain, and the explicit order
// annotation override from explicit_order.

#include "TestUtils.hpp"

#include "perf/Sim/ExecutionView.h"
#include "perf/Sim/ScheduleView.h"

using namespace mlir;
using namespace perf;
using namespace perftest;

namespace {

class ScheduleViewTest : public SimViewTestBase {
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

TEST_F(ScheduleViewTest, DefaultOrderIsProgramOrderPerResource) {
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
TEST_F(SimViewTestBase, ExplicitOrderAnnotationOverridesProgramOrder) {
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
