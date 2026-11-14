// BufferProvenanceTest.cpp - Unit tests for buffer provenance analysis.

#include "perf/Analysis/BufferProvenance.h"

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

class BufferProvenanceTest : public ::testing::Test {
protected:
  void SetUp() override {
    context.loadDialect<PerfDialect>();
    context.loadDialect<func::FuncDialect>();
    context.loadDialect<memref::MemRefDialect>();
  }

  MLIRContext context;
  OwningOpRef<ModuleOp> module;

  func::FuncOp parseFunc(StringRef text) {
    module = parseSourceString<ModuleOp>(text, &context);
    if (!module) {
      llvm::errs() << "parseSourceString failed\n";
      return nullptr;
    }
    return module->lookupSymbol<func::FuncOp>("test");
  }

  /// Get the i-th op result in the entry block (skipping terminators).
  Value getOpResult(func::FuncOp func, unsigned opIndex,
                    unsigned resultIndex = 0) {
    auto &block = func.getBody().front();
    auto it = block.begin();
    std::advance(it, opIndex);
    return (*it).getResult(resultIndex);
  }
};

//===----------------------------------------------------------------------===//
// perf.alloc: root buffer, trivial provenance
//===----------------------------------------------------------------------===//

TEST_F(BufferProvenanceTest, AllocIsItsOwnRoot) {
  const char *text = R"mlir(
module {
  func.func @test() {
    %buf = perf.alloc : memref<512xi8, #perf.space<"spad", 0>>
    return
  }
}
)mlir";

  auto func = parseFunc(text);
  ASSERT_TRUE(func);

  Value buf = getOpResult(func, 0);
  auto prov = resolve(buf);
  ASSERT_TRUE(succeeded(prov));

  EXPECT_TRUE(isa<AllocOp>(prov->root));
  EXPECT_EQ(prov->byteOffset, 0);
  EXPECT_EQ(prov->byteSize, 512);
  EXPECT_FALSE(prov->isHost);
}

//===----------------------------------------------------------------------===//
// func block argument: host-side root
//===----------------------------------------------------------------------===//

TEST_F(BufferProvenanceTest, BlockArgIsHostRoot) {
  const char *text = R"mlir(
module {
  func.func @test(%arg0: memref<512xi8>) {
    return
  }
}
)mlir";

  auto func = parseFunc(text);
  ASSERT_TRUE(func);

  Value arg = func.getArgument(0);
  auto prov = resolve(arg);
  ASSERT_TRUE(succeeded(prov));

  EXPECT_TRUE(prov->isHost);
  EXPECT_EQ(prov->byteOffset, 0);
  EXPECT_EQ(prov->byteSize, 512);
}

TEST_F(BufferProvenanceTest, DistinctBlockArgsAreDistinctRoots) {
  const char *text = R"mlir(
module {
  func.func @test(%arg0: memref<512xi8>, %arg1: memref<512xi8>) {
    return
  }
}
)mlir";

  auto func = parseFunc(text);
  ASSERT_TRUE(func);

  auto p0 = resolve(func.getArgument(0));
  auto p1 = resolve(func.getArgument(1));
  ASSERT_TRUE(succeeded(p0));
  ASSERT_TRUE(succeeded(p1));

  EXPECT_FALSE(conflict(*p0, *p1));
}

//===----------------------------------------------------------------------===//
// conflict(): distinct roots never conflict
//===----------------------------------------------------------------------===//

TEST_F(BufferProvenanceTest, DistinctAllocRootsDoNotConflict) {
  const char *text = R"mlir(
module {
  func.func @test() {
    %a = perf.alloc : memref<512xi8, #perf.space<"spad", 0>>
    %b = perf.alloc : memref<512xi8, #perf.space<"spad", 1>>
    return
  }
}
)mlir";

  auto func = parseFunc(text);
  ASSERT_TRUE(func);

  auto provA = resolve(getOpResult(func, 0));
  auto provB = resolve(getOpResult(func, 1));
  ASSERT_TRUE(succeeded(provA));
  ASSERT_TRUE(succeeded(provB));

  EXPECT_FALSE(conflict(*provA, *provB));
}

TEST_F(BufferProvenanceTest, SameRootFullRangeConflicts) {
  const char *text = R"mlir(
module {
  func.func @test() {
    %a = perf.alloc : memref<512xi8, #perf.space<"spad", 0>>
    return
  }
}
)mlir";

  auto func = parseFunc(text);
  ASSERT_TRUE(func);

  auto provA = resolve(getOpResult(func, 0));
  ASSERT_TRUE(succeeded(provA));

  EXPECT_TRUE(conflict(*provA, *provA));
}

//===----------------------------------------------------------------------===//
// memref.subview: static offset, narrowed range
//===----------------------------------------------------------------------===//

TEST_F(BufferProvenanceTest, SubviewTracksOffsetAndSize) {
  const char *text = R"mlir(
module {
  func.func @test() {
    %buf = perf.alloc : memref<512xi8, #perf.space<"spad", 0>>
    %sub = memref.subview %buf[256] [256] [1]
           : memref<512xi8, #perf.space<"spad", 0>> to memref<256xi8, strided<[1], offset: 256>, #perf.space<"spad", 0>>
    return
  }
}
)mlir";

  auto func = parseFunc(text);
  ASSERT_TRUE(func);

  Value sub = getOpResult(func, 1);
  auto prov = resolve(sub);
  ASSERT_TRUE(succeeded(prov));

  EXPECT_TRUE(isa<AllocOp>(prov->root));
  EXPECT_EQ(prov->byteOffset, 256);
  EXPECT_EQ(prov->byteSize, 256);

  // subview is a subset of the root buffer: must conflict with the root.
  auto provBuf = resolve(getOpResult(func, 0));
  ASSERT_TRUE(succeeded(provBuf));
  EXPECT_TRUE(conflict(*provBuf, *prov));
}

TEST_F(BufferProvenanceTest, DisjointSubviewsDoNotConflict) {
  const char *text = R"mlir(
module {
  func.func @test() {
    %buf = perf.alloc : memref<512xi8, #perf.space<"spad", 0>>
    %s0 = memref.subview %buf[0] [256] [1]
          : memref<512xi8, #perf.space<"spad", 0>> to memref<256xi8, strided<[1], offset: 0>, #perf.space<"spad", 0>>
    %s1 = memref.subview %buf[256] [256] [1]
          : memref<512xi8, #perf.space<"spad", 0>> to memref<256xi8, strided<[1], offset: 256>, #perf.space<"spad", 0>>
    return
  }
}
)mlir";

  auto func = parseFunc(text);
  ASSERT_TRUE(func);

  auto p0 = resolve(getOpResult(func, 1));
  auto p1 = resolve(getOpResult(func, 2));
  ASSERT_TRUE(succeeded(p0));
  ASSERT_TRUE(succeeded(p1));

  EXPECT_FALSE(conflict(*p0, *p1));
}

TEST_F(BufferProvenanceTest, OverlappingSubviewsConflict) {
  const char *text = R"mlir(
module {
  func.func @test() {
    %buf = perf.alloc : memref<512xi8, #perf.space<"spad", 0>>
    %s0 = memref.subview %buf[0] [512] [1]
          : memref<512xi8, #perf.space<"spad", 0>> to memref<512xi8, strided<[1], offset: 0>, #perf.space<"spad", 0>>
    %s1 = memref.subview %buf[256] [256] [1]
          : memref<512xi8, #perf.space<"spad", 0>> to memref<256xi8, strided<[1], offset: 256>, #perf.space<"spad", 0>>
    return
  }
}
)mlir";

  auto func = parseFunc(text);
  ASSERT_TRUE(func);

  auto p0 = resolve(getOpResult(func, 1));
  auto p1 = resolve(getOpResult(func, 2));
  ASSERT_TRUE(succeeded(p0));
  ASSERT_TRUE(succeeded(p1));

  EXPECT_TRUE(conflict(*p0, *p1));
}

TEST_F(BufferProvenanceTest, SubviewRootIsAllocNotSubview) {
  // Root must be the alloc, not the intermediate subview op.
  const char *text = R"mlir(
module {
  func.func @test() {
    %buf = perf.alloc : memref<512xi8, #perf.space<"spad", 0>>
    %sub = memref.subview %buf[256] [256] [1]
           : memref<512xi8, #perf.space<"spad", 0>> to memref<256xi8, strided<[1], offset: 256>, #perf.space<"spad", 0>>
    return
  }
}
)mlir";

  auto func = parseFunc(text);
  ASSERT_TRUE(func);

  auto provSub = resolve(getOpResult(func, 1));
  auto provBuf = resolve(getOpResult(func, 0));
  ASSERT_TRUE(succeeded(provSub));
  ASSERT_TRUE(succeeded(provBuf));

  EXPECT_EQ(provSub->root, provBuf->root);
}

//===----------------------------------------------------------------------===//
// Multi-element subview: offset scaled by element size
//===----------------------------------------------------------------------===//

TEST_F(BufferProvenanceTest, SubviewOffsetScaledByElementSize) {
  // i32 elements: offset 64 elements = 256 bytes.
  const char *text = R"mlir(
module {
  func.func @test() {
    %buf = perf.alloc : memref<256xi32, #perf.space<"acc", 0>>
    %sub = memref.subview %buf[64] [64] [1]
           : memref<256xi32, #perf.space<"acc", 0>> to memref<64xi32, strided<[1], offset: 64>, #perf.space<"acc", 0>>
    return
  }
}
)mlir";

  auto func = parseFunc(text);
  ASSERT_TRUE(func);

  auto prov = resolve(getOpResult(func, 1));
  ASSERT_TRUE(succeeded(prov));

  EXPECT_EQ(prov->byteOffset, 64 * 4);
  EXPECT_EQ(prov->byteSize, 64 * 4);
}

//===----------------------------------------------------------------------===//
// Chained subview: view of view
//===----------------------------------------------------------------------===//

TEST_F(BufferProvenanceTest, ChainedSubviewAccumulatesOffsets) {
  // buf[256..512) -> sub[64..192) => root range [320, 448)
  const char *text = R"mlir(
module {
  func.func @test() {
    %buf = perf.alloc : memref<512xi8, #perf.space<"spad", 0>>
    %s0 = memref.subview %buf[256] [256] [1]
          : memref<512xi8, #perf.space<"spad", 0>> to memref<256xi8, strided<[1], offset: 256>, #perf.space<"spad", 0>>
    %s1 = memref.subview %s0[64] [128] [1]
          : memref<256xi8, strided<[1], offset: 256>, #perf.space<"spad", 0>> to memref<128xi8, strided<[1], offset: 320>, #perf.space<"spad", 0>>
    return
  }
}
)mlir";

  auto func = parseFunc(text);
  ASSERT_TRUE(func);

  auto prov = resolve(getOpResult(func, 2));
  ASSERT_TRUE(succeeded(prov));

  EXPECT_TRUE(isa<AllocOp>(prov->root));
  EXPECT_EQ(prov->byteOffset, 320);
  EXPECT_EQ(prov->byteSize, 128);
}

TEST_F(BufferProvenanceTest, ChainedSubviewConflictsWithRoot) {
  // Chained subview overlaps with the root's full range.
  const char *text = R"mlir(
module {
  func.func @test() {
    %buf = perf.alloc : memref<512xi8, #perf.space<"spad", 0>>
    %s0 = memref.subview %buf[256] [256] [1]
          : memref<512xi8, #perf.space<"spad", 0>> to memref<256xi8, strided<[1], offset: 256>, #perf.space<"spad", 0>>
    return
  }
}
)mlir";

  auto func = parseFunc(text);
  ASSERT_TRUE(func);

  auto provBuf = resolve(getOpResult(func, 0));
  auto provSub = resolve(getOpResult(func, 1));
  ASSERT_TRUE(succeeded(provBuf));
  ASSERT_TRUE(succeeded(provSub));

  EXPECT_TRUE(conflict(*provBuf, *provSub));
}

//===----------------------------------------------------------------------===//
// Unsupported constructs: must fail (verifier rejects)
//===----------------------------------------------------------------------===//

TEST_F(BufferProvenanceTest, DynamicOffsetSubviewFails) {
  const char *text = R"mlir(
module {
  func.func @test(%i: index) {
    %buf = perf.alloc : memref<512xi8, #perf.space<"spad", 0>>
    %sub = memref.subview %buf[%i] [256] [1]
           : memref<512xi8, #perf.space<"spad", 0>> to memref<256xi8, strided<[1], offset: ?>, #perf.space<"spad", 0>>
    return
  }
}
)mlir";

  auto func = parseFunc(text);
  ASSERT_TRUE(func);

  auto prov = resolve(getOpResult(func, 1));
  EXPECT_TRUE(failed(prov));
}

TEST_F(BufferProvenanceTest, AllocResultUsedDirectlyIsFullRange) {
  // Sanity: alloc itself spans [0, allocSize).
  const char *text = R"mlir(
module {
  func.func @test() {
    %buf = perf.alloc : memref<128xi8, #perf.space<"spad", 3>>
    return
  }
}
)mlir";

  auto func = parseFunc(text);
  ASSERT_TRUE(func);

  auto prov = resolve(getOpResult(func, 0));
  ASSERT_TRUE(succeeded(prov));

  EXPECT_EQ(prov->byteOffset, 0);
  EXPECT_EQ(prov->byteSize, 128);
}
