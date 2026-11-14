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

class ReachabilityVerifierTest : public ::testing::Test {
protected:
  void SetUp() override {
    context.loadDialect<PerfDialect>();
    context.loadDialect<func::FuncDialect>();
    context.loadDialect<memref::MemRefDialect>();
  }

  MLIRContext context;
  OwningOpRef<ModuleOp> module;

  ModuleOp parseModule(StringRef text) {
    module = parseSourceString<ModuleOp>(text, &context);
    return module ? *module : nullptr;
  }
};

//===----------------------------------------------------------------------===//
// Positive: all ops access reachable spaces
//===----------------------------------------------------------------------===//

TEST_F(ReachabilityVerifierTest, ValidMatmulPasses) {
  const char *text = R"mlir(
module {
  func.func @matmul_impl(%A: memref<512xi8>, %B: memref<512xi8>, %C: memref<256xi32>) {
    %free = perf.token.init

    %spad_a0 = perf.alloc : memref<512xi8, #perf.space<"spad", 0>>
    %spad_b  = perf.alloc : memref<512xi8, #perf.space<"spad", 2>>
    %acc_c0  = perf.alloc : memref<256xi32, #perf.space<"acc", 0>>

    %la = perf.dma_load %A, %spad_a0 [%free] {
      bytes = 512,
      resource = #perf.res<"dma_read", 0>
    } : (memref<512xi8>, memref<512xi8, #perf.space<"spad", 0>>, !perf.token) -> !perf.token

    %lb = perf.dma_load %B, %spad_b [%free] {
      bytes = 512,
      resource = #perf.res<"dma_read", 0>
    } : (memref<512xi8>, memref<512xi8, #perf.space<"spad", 2>>, !perf.token) -> !perf.token

    %p0 = perf.preload %acc_c0 {
      role = "output", m = 16, n = 16, init = "zero",
      resource = #perf.res<"array", 0>
    } : (memref<256xi32, #perf.space<"acc", 0>>) -> !perf.token

    %c0 = perf.compute %spad_a0, %spad_b, %acc_c0 [%la, %lb, %p0] {
      m = 16, n = 16, k = 32,
      accumulate = false, writeback = true,
      resource = #perf.res<"array", 10>
    } : (memref<512xi8, #perf.space<"spad", 0>>,
         memref<512xi8, #perf.space<"spad", 2>>,
         memref<256xi32, #perf.space<"acc", 0>>,
         !perf.token, !perf.token, !perf.token) -> !perf.token

    %st0 = perf.dma_store %acc_c0, %C [%c0] {
      bytes = 1024,
      resource = #perf.res<"dma_write", 0>
    } : (memref<256xi32, #perf.space<"acc", 0>>, memref<256xi32>, !perf.token) -> !perf.token

    perf.wait [%st0]
    return
  }
}
)mlir";

  auto mod = parseModule(text);
  ASSERT_TRUE(mod);

  EXPECT_TRUE(succeeded(verifyReachability(mod)));
}

//===----------------------------------------------------------------------===//
// Note: op verifiers already reject host-DRAM operands for compute/preload.
// R1's value is for perf.prim and future resource kinds without op-level
// verification. We test R1's pass-through for valid programs and its
// rejection of unknown resource kinds.
//===----------------------------------------------------------------------===//

TEST_F(ReachabilityVerifierTest, ComputeWithOnChipOperandsPasses) {
  const char *text = R"mlir(
module {
  func.func @test() {
    %spad_a = perf.alloc : memref<512xi8, #perf.space<"spad", 0>>
    %spad_b = perf.alloc : memref<512xi8, #perf.space<"spad", 1>>
    %acc_c  = perf.alloc : memref<256xi32, #perf.space<"acc", 0>>

    %c0 = perf.compute %spad_a, %spad_b, %acc_c [] {
      m = 16, n = 16, k = 32,
      accumulate = false, writeback = true,
      resource = #perf.res<"array", 0>
    } : (memref<512xi8, #perf.space<"spad", 0>>,
         memref<512xi8, #perf.space<"spad", 1>>,
         memref<256xi32, #perf.space<"acc", 0>>) -> !perf.token

    perf.wait [%c0]
    return
  }
}
)mlir";

  auto mod = parseModule(text);
  ASSERT_TRUE(mod);

  EXPECT_TRUE(succeeded(verifyReachability(mod)));
}

TEST_F(ReachabilityVerifierTest, PreloadWithOnChipOperandPasses) {
  const char *text = R"mlir(
module {
  func.func @test() {
    %acc_c = perf.alloc : memref<256xi32, #perf.space<"acc", 0>>

    %p0 = perf.preload %acc_c {
      role = "output", m = 16, n = 16, init = "zero",
      resource = #perf.res<"array", 0>
    } : (memref<256xi32, #perf.space<"acc", 0>>) -> !perf.token

    perf.wait [%p0]
    return
  }
}
)mlir";

  auto mod = parseModule(text);
  ASSERT_TRUE(mod);

  EXPECT_TRUE(succeeded(verifyReachability(mod)));
}

//===----------------------------------------------------------------------===//
// dma_load src on-chip is fine for R1 (direction is op verifier's job)
//===----------------------------------------------------------------------===//

TEST_F(ReachabilityVerifierTest, DmaLoadSrcOnChipPassesR1) {
  const char *text = R"mlir(
module {
  func.func @test() {
    %spad_a = perf.alloc : memref<512xi8, #perf.space<"spad", 0>>
    %spad_b = perf.alloc : memref<512xi8, #perf.space<"spad", 1>>

    %la = perf.dma_load %spad_a, %spad_b [] {
      bytes = 512,
      resource = #perf.res<"dma_read", 0>
    } : (memref<512xi8, #perf.space<"spad", 0>>,
         memref<512xi8, #perf.space<"spad", 1>>) -> !perf.token

    perf.wait [%la]
    return
  }
}
)mlir";

  auto mod = parseModule(text);
  ASSERT_TRUE(mod);

  // R1 only checks connectivity, not direction. dma_read can access spad.
  EXPECT_TRUE(succeeded(verifyReachability(mod)));
}

//===----------------------------------------------------------------------===//
// dma_store dst on-chip is fine for R1
//===----------------------------------------------------------------------===//

TEST_F(ReachabilityVerifierTest, DmaStoreDstOnChipPassesR1) {
  const char *text = R"mlir(
module {
  func.func @test() {
    %acc_c = perf.alloc : memref<256xi32, #perf.space<"acc", 0>>
    %spad  = perf.alloc : memref<256xi32, #perf.space<"spad", 0>>

    %st0 = perf.dma_store %acc_c, %spad [] {
      bytes = 1024,
      resource = #perf.res<"dma_write", 0>
    } : (memref<256xi32, #perf.space<"acc", 0>>,
         memref<256xi32, #perf.space<"spad", 0>>) -> !perf.token

    perf.wait [%st0]
    return
  }
}
)mlir";

  auto mod = parseModule(text);
  ASSERT_TRUE(mod);

  // R1 only checks connectivity, not direction.
  EXPECT_TRUE(succeeded(verifyReachability(mod)));
}

//===----------------------------------------------------------------------===//
// Negative: unknown resource kind
//===----------------------------------------------------------------------===//

TEST_F(ReachabilityVerifierTest, UnknownResourceKindFails) {
  const char *text = R"mlir(
module {
  func.func @test(%A: memref<512xi8>) {
    %spad = perf.alloc : memref<512xi8, #perf.space<"spad", 0>>

    %la = perf.dma_load %A, %spad [] {
      bytes = 512,
      resource = #perf.res<"nonexistent", 0>
    } : (memref<512xi8>, memref<512xi8, #perf.space<"spad", 0>>) -> !perf.token

    perf.wait [%la]
    return
  }
}
)mlir";

  auto mod = parseModule(text);
  ASSERT_TRUE(mod);

  EXPECT_TRUE(failed(verifyReachability(mod)));
}
