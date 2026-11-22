// ReachabilityVerifierTest.cpp - Unit tests for R1 reachability verifier.

#include "perf/Analysis/ReachabilityVerifier.h"

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

// Shared prior (calibration state) for building TargetModels in tests.
static constexpr const char *kPrior = R"mlir(
module attributes {perf.prior = #perf.prior<
  version = 5,
  target_name = "test_prior",
  calibration = #perf.calibration<
    extra_fill_cycles = 0, extra_drain_cycles = 0,
    extra_preload_cycles = 0, extra_writeback_cycles = 0,
    host_issue_cycles = 0
  >
>} {}
)mlir";

// Baseline design: 1 DMA read port, 1 DMA write port, 1 array, shared bus.
static constexpr const char *kDesign = R"mlir(
module attributes {perf.design = #perf.design<
  version = 5,
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
  dma_read_ports = 1,
  dma_write_ports = 1,
  arrays = 1,
  dma_channel = #perf.dma<bus_bytes_per_cycle = 16, fixed_latency_cycles = 20,
                          max_outstanding = 8, burst_bytes = 64>,
  array_uarch = #perf.array<topology = "systolic",
                            mesh_rows = 16, mesh_cols = 16,
                            tile_rows = 1, tile_cols = 1,
                            pe_latency_cycles = 1,
                            input_skew = true,
                            inter_tile_pipelined = true>,
  share_dram_bus = true
>} {}
)mlir";

class ReachabilityVerifierTest : public ::testing::Test {
protected:
  void SetUp() override {
    context.loadDialect<PerfDialect>();
    context.loadDialect<func::FuncDialect>();
    context.loadDialect<memref::MemRefDialect>();

    auto priorMod = parseSourceString<ModuleOp>(kPrior, &context);
    ASSERT_TRUE(priorMod);
    auto priorAttr = getPriorAttr(*priorMod);
    ASSERT_TRUE(priorAttr);

    auto designMod = parseSourceString<ModuleOp>(kDesign, &context);
    ASSERT_TRUE(designMod);
    auto designAttr = getDesignAttr(*designMod);
    ASSERT_TRUE(designAttr);

    auto modelOr = TargetModel::build(designAttr, priorAttr);
    ASSERT_TRUE(succeeded(modelOr));
    target = std::make_unique<TargetModel>(std::move(*modelOr));
  }

  MLIRContext context;
  std::unique_ptr<TargetModel> target;

  DesignAttr getDesignAttr(ModuleOp mod) {
    return llvm::dyn_cast_if_present<DesignAttr>(
        mod->getAttr("perf.design"));
  }

  PriorAttr getPriorAttr(ModuleOp mod) {
    return llvm::dyn_cast_if_present<PriorAttr>(
        mod->getAttr("perf.prior"));
  }

  LogicalResult verify(StringRef programText) {
    auto mod = parseSourceString<ModuleOp>(programText, &context);
    if (!mod)
      return failure();
    return verifyReachability(*mod, *target);
  }
};

//===----------------------------------------------------------------------===//
// Positive: the canonical matmul fixture passes
//===----------------------------------------------------------------------===//

TEST_F(ReachabilityVerifierTest, ValidMatmulPasses) {
  const char *text = R"mlir(
module {
  func.func @matmul_impl(%A: memref<512xi8>, %B: memref<512xi8>, %C: memref<256xi32>) {
    %free = perf.token.init

    %spad_a0 = perf.alloc : memref<512xi8, #perf.space<"spad", 0, 32>>
    %spad_b  = perf.alloc : memref<512xi8, #perf.space<"spad", 32, 32>>
    %acc_c0  = perf.alloc : memref<256xi32, #perf.space<"acc", 0, 16>>

    %la = perf.dma_load %A, %spad_a0 [%free] {
      bytes = 512,
      resource = #perf.res<"dma_read", 0>
    } : (memref<512xi8>, memref<512xi8, #perf.space<"spad", 0, 32>>, !perf.token) -> !perf.token

    %lb = perf.dma_load %B, %spad_b [%free] {
      bytes = 512,
      resource = #perf.res<"dma_read", 0>
    } : (memref<512xi8>, memref<512xi8, #perf.space<"spad", 32, 32>>, !perf.token) -> !perf.token

    %p0 = perf.array_init {
      m = 16, n = 16,
      resource = #perf.res<"array", 0>
    } : () -> !perf.token

    %c0 = perf.compute %spad_a0, %spad_b, %acc_c0 [%la, %lb, %p0] {
      m = 16, n = 16, k = 32,
      writeback = true,
      resource = #perf.res<"array", 0>
    } : (memref<512xi8, #perf.space<"spad", 0, 32>>,
         memref<512xi8, #perf.space<"spad", 32, 32>>,
         memref<256xi32, #perf.space<"acc", 0, 16>>,
         !perf.token, !perf.token, !perf.token) -> !perf.token

    %st0 = perf.dma_store %acc_c0, %C [%c0] {
      bytes = 1024,
      resource = #perf.res<"dma_write", 0>
    } : (memref<256xi32, #perf.space<"acc", 0, 16>>, memref<256xi32>, !perf.token) -> !perf.token

    perf.wait [%st0]
    return
  }
}
)mlir";

  EXPECT_TRUE(succeeded(verify(text)));
}

// array_init with an operand reads old partial sums from the acc space.
TEST_F(ReachabilityVerifierTest, ArrayInitWithOperandPasses) {
  const char *text = R"mlir(
module {
  func.func @t(%acc: memref<256xi32, #perf.space<"acc", 0, 16>>) {
    %t = perf.token.init
    %i0 = perf.array_init %acc [%t] {
      m = 16, n = 16,
      resource = #perf.res<"array", 0>
    } : (memref<256xi32, #perf.space<"acc", 0, 16>>, !perf.token) -> !perf.token
    return
  }
}
)mlir";

  EXPECT_TRUE(succeeded(verify(text)));
}

//===----------------------------------------------------------------------===//
// Negative: placement resource must exist in the contract
//===----------------------------------------------------------------------===//

TEST_F(ReachabilityVerifierTest, UnknownExecutorKindFails) {
  const char *text = R"mlir(
module {
  func.func @t(%A: memref<512xi8>, %B: memref<512xi8, #perf.space<"spad", 0, 32>>) {
    %t = perf.token.init
    %l = perf.dma_load %A, %B [%t] {
      bytes = 512,
      resource = #perf.res<"dma_rade", 0>
    } : (memref<512xi8>, memref<512xi8, #perf.space<"spad", 0, 32>>, !perf.token) -> !perf.token
    return
  }
}
)mlir";

  EXPECT_TRUE(failed(verify(text)));
}

TEST_F(ReachabilityVerifierTest, ExecutorIndexOutOfRangeFails) {
  const char *text = R"mlir(
module {
  func.func @t(%A: memref<512xi8>, %B: memref<512xi8, #perf.space<"spad", 0, 32>>) {
    %t = perf.token.init
    %l = perf.dma_load %A, %B [%t] {
      bytes = 512,
      resource = #perf.res<"dma_read", 7>
    } : (memref<512xi8>, memref<512xi8, #perf.space<"spad", 0, 32>>, !perf.token) -> !perf.token
    return
  }
}
)mlir";

  EXPECT_TRUE(failed(verify(text)));
}

// Links are internal: they cannot be referenced by IR placement.
TEST_F(ReachabilityVerifierTest, LinkKindRejectedAsPlacement) {
  const char *text = R"mlir(
module {
  func.func @t(%A: memref<512xi8>, %B: memref<512xi8, #perf.space<"spad", 0, 32>>) {
    %t = perf.token.init
    %l = perf.dma_load %A, %B [%t] {
      bytes = 512,
      resource = #perf.res<"dram_bus", 0>
    } : (memref<512xi8>, memref<512xi8, #perf.space<"spad", 0, 32>>, !perf.token) -> !perf.token
    return
  }
}
)mlir";

  EXPECT_TRUE(failed(verify(text)));
}

//===----------------------------------------------------------------------===//
// Negative: operand space must be declared and reachable
//===----------------------------------------------------------------------===//

TEST_F(ReachabilityVerifierTest, UnknownMemorySpaceFails) {
  const char *text = R"mlir(
module {
  func.func @t(%A: memref<512xi8>, %B: memref<512xi8, #perf.space<"typo_spad", 0, 32>>) {
    %t = perf.token.init
    %l = perf.dma_load %A, %B [%t] {
      bytes = 512,
      resource = #perf.res<"dma_read", 0>
    } : (memref<512xi8>, memref<512xi8, #perf.space<"typo_spad", 0, 32>>, !perf.token) -> !perf.token
    return
  }
}
)mlir";

  EXPECT_TRUE(failed(verify(text)));
}

// The array is not a DMA bridge: it must not touch host memory.
TEST_F(ReachabilityVerifierTest, ArrayCannotAccessHostMemory) {
  const char *text = R"mlir(
module {
  func.func @t(%A: memref<512xi8>,
               %B: memref<512xi8, #perf.space<"spad", 0, 32>>,
               %C: memref<256xi32, #perf.space<"acc", 0, 16>>) {
    %t = perf.token.init
    %c0 = perf.compute %A, %B, %C [%t] {
      m = 16, n = 16, k = 32,
      writeback = true,
      resource = #perf.res<"array", 0>
    } : (memref<512xi8>,
         memref<512xi8, #perf.space<"spad", 0, 32>>,
         memref<256xi32, #perf.space<"acc", 0, 16>>,
         !perf.token) -> !perf.token
    return
  }
}
)mlir";

  EXPECT_TRUE(failed(verify(text)));
}
