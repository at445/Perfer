// ExecutionViewTest.cpp - Unit tests for the simulator's index layer.

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

#include "gtest/gtest.h"

using namespace mlir;
using namespace perf;

// A small program exercising all four executor ops with a token chain.
static constexpr const char *kProgram = R"mlir(
module {
  func.func @main(%A: memref<512xi8>, %B: memref<512xi8>, %C: memref<256xi32>,
                  %SA: memref<512xi8, #perf.space<"spad", 0, 32>>,
                  %SB: memref<512xi8, #perf.space<"spad", 32, 32>>,
                  %ACC: memref<256xi32, #perf.space<"acc", 0, 16>>) {
    %free = perf.token.init

    %la = perf.dma_load %A, %SA [%free] {
      bytes = 512,
      resource = #perf.res<"dma_read", 0>
    } : (memref<512xi8>, memref<512xi8, #perf.space<"spad", 0, 32>>, !perf.token) -> !perf.token

    %lb = perf.dma_load %B, %SB [%free] {
      bytes = 512,
      resource = #perf.res<"dma_read", 0>
    } : (memref<512xi8>, memref<512xi8, #perf.space<"spad", 32, 32>>, !perf.token) -> !perf.token

    %i0 = perf.array_init {
      m = 16, n = 16,
      resource = #perf.res<"array", 0>
    } : () -> !perf.token

    %c0 = perf.compute %SA, %SB, %ACC [%la, %lb, %i0] {
      m = 16, n = 16, k = 32,
      writeback = true,
      resource = #perf.res<"array", 0>
    } : (memref<512xi8, #perf.space<"spad", 0, 32>>,
         memref<512xi8, #perf.space<"spad", 32, 32>>,
         memref<256xi32, #perf.space<"acc", 0, 16>>,
         !perf.token, !perf.token, !perf.token) -> !perf.token

    %st = perf.dma_store %ACC, %C [%c0] {
      bytes = 1024,
      resource = #perf.res<"dma_write", 0>
    } : (memref<256xi32, #perf.space<"acc", 0, 16>>, memref<256xi32>, !perf.token) -> !perf.token

    perf.fence [%st]
    return
  }
}
)mlir";

class SimIndexTest : public ::testing::Test {
protected:
  void SetUp() override {
    context.loadDialect<PerfDialect>();
    context.loadDialect<func::FuncDialect>();
    context.loadDialect<memref::MemRefDialect>();
    module = parseSourceString<ModuleOp>(kProgram, &context);
    ASSERT_TRUE(module);
    view = ExecutionView::build(*module);
  }

  MLIRContext context;
  OwningOpRef<ModuleOp> module;
  ExecutionView view;
};

//===----------------------------------------------------------------------===//
// ExecutionView
//===----------------------------------------------------------------------===//

TEST_F(SimIndexTest, IndexesAllExecutorOpsInProgramOrder) {
  ASSERT_EQ(view.ops().size(), 5u); // la, lb, i0, c0, st

  // Ops are in block program order; ids are dense and stable.
  for (OpId id = 0; id < view.ops().size(); ++id) {
    Operation *op = view.opOf(id);
    EXPECT_EQ(view.idOf(op), id);
  }
}

TEST_F(SimIndexTest, PlacementParsedFromResourceAttr) {
  EXPECT_EQ(view.placementOf(view.opOf(0)), ResourceId("dma_read", 0));
  EXPECT_EQ(view.placementOf(view.opOf(1)), ResourceId("dma_read", 0));
  EXPECT_EQ(view.placementOf(view.opOf(2)), ResourceId("array", 0));
  EXPECT_EQ(view.placementOf(view.opOf(3)), ResourceId("array", 0));
  EXPECT_EQ(view.placementOf(view.opOf(4)), ResourceId("dma_write", 0));
}

TEST_F(SimIndexTest, TokenProducerCoversExecutorResults) {
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

TEST_F(SimIndexTest, TokenEdgesFormExpectedGraph) {
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

TEST_F(SimIndexTest, ReadyTracksDoneMarks) {
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

TEST_F(SimIndexTest, DefaultOrderIsProgramOrderPerResource) {
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

TEST_F(SimIndexTest, ExplicitOrderAnnotationOverridesProgramOrder) {
  const char *text = R"mlir(
module {
  func.func @t(%A: memref<512xi8>, %B: memref<512xi8>,
               %SA: memref<512xi8, #perf.space<"spad", 0, 32>>,
               %SB: memref<512xi8, #perf.space<"spad", 32, 32>>) {
    %free = perf.token.init
    %la = perf.dma_load %A, %SA [%free] {
      bytes = 512, resource = #perf.res<"dma_read", 0>, order = 2
    } : (memref<512xi8>, memref<512xi8, #perf.space<"spad", 0, 32>>, !perf.token) -> !perf.token
    %lb = perf.dma_load %B, %SB [%free] {
      bytes = 512, resource = #perf.res<"dma_read", 0>, order = 1
    } : (memref<512xi8>, memref<512xi8, #perf.space<"spad", 32, 32>>, !perf.token) -> !perf.token
    return
  }
}
)mlir";

  auto mod2 = parseSourceString<ModuleOp>(text, &context);
  ASSERT_TRUE(mod2);
  auto view2 = ExecutionView::build(*mod2);
  auto sched = ScheduleView::build(view2);

  // Both ops annotated: order=1 (lb, id 1) precedes order=2 (la, id 0).
  ResourceId dmaRead{"dma_read", 0};
  EXPECT_EQ(sched.head(dmaRead), 1u);
  sched.advance(dmaRead);
  EXPECT_EQ(sched.head(dmaRead), 0u);
}
