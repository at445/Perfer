// BufferProvenance.cpp - Static buffer identity and alias analysis.

#include "perf/Analysis/BufferProvenance.h"

#include "perf/Dialect/PerfOps.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"

using namespace mlir;

namespace perf {

namespace {

/// Compute the total byte size of a static-shaped memref.
FailureOr<int64_t> getMemRefByteSize(MemRefType type) {
  if (!type.hasStaticShape())
    return failure();
  int64_t numElements = 1;
  for (int64_t dim : type.getShape())
    numElements *= dim;
  unsigned elemBits = type.getElementTypeBitWidth();
  if (elemBits % 8 != 0)
    return failure();
  return numElements * (elemBits / 8);
}

/// Compute the byte size of one element.
FailureOr<int64_t> getElementByteSize(MemRefType type) {
  unsigned elemBits = type.getElementTypeBitWidth();
  if (elemBits % 8 != 0)
    return failure();
  return static_cast<int64_t>(elemBits / 8);
}

} // namespace

llvm::FailureOr<Provenance> resolve(Value v) {
  auto memrefType = dyn_cast<MemRefType>(v.getType());
  if (!memrefType)
    return failure();

  // Base case: block argument -> host-side root.
  // Each argument is a distinct root; use the parent func + arg index.
  if (auto blockArg = dyn_cast<BlockArgument>(v)) {
    auto size = getMemRefByteSize(memrefType);
    if (failed(size))
      return failure();
    return Provenance{blockArg.getOwner()->getParentOp(),
                      static_cast<int>(blockArg.getArgNumber()), 0, *size,
                      true};
  }

  Operation *defOp = v.getDefiningOp();
  if (!defOp)
    return failure();

  // Base case: perf.alloc -> on-chip root.
  if (auto allocOp = dyn_cast<AllocOp>(defOp)) {
    auto size = getMemRefByteSize(memrefType);
    if (failed(size))
      return failure();
    return Provenance{allocOp.getOperation(), -1, 0, *size, false};
  }

  // Recursive case: memref.subview with static offsets/sizes/strides.
  if (auto subviewOp = dyn_cast<memref::SubViewOp>(defOp)) {
    // Only support 1-D subviews in Phase 1.
    if (subviewOp.getSourceType().getRank() != 1)
      return failure();

    // Offsets must be static (no dynamic offset operands).
    if (!subviewOp.getOffsets().empty())
      return failure();

    auto staticOffsets = subviewOp.getStaticOffsets();
    auto staticSizes = subviewOp.getStaticSizes();
    if (staticOffsets.size() != 1 || staticSizes.size() != 1)
      return failure();

    int64_t elemOffset = staticOffsets[0];
    int64_t numElements = staticSizes[0];
    if (elemOffset == ShapedType::kDynamic ||
        numElements == ShapedType::kDynamic)
      return failure();

    auto parentProv = resolve(subviewOp.getSource());
    if (failed(parentProv))
      return failure();

    auto elemSize = getElementByteSize(memrefType);
    if (failed(elemSize))
      return failure();

    return Provenance{parentProv->root, parentProv->argIndex,
                      parentProv->byteOffset + elemOffset * (*elemSize),
                      numElements * (*elemSize), parentProv->isHost};
  }

  //Todo: Anything else (phi, iter_args, unknown ops) is unsupported in Phase 1.
  return failure();
}

bool conflict(const Provenance &u, const Provenance &v) {
  if (u.isHost != v.isHost)
    return false;
  if (u.isHost) {
    // Both are block arguments: same func and same arg index means same root.
    if (u.root != v.root || u.argIndex != v.argIndex)
      return false;
  } else {
    // Both are allocs: root must be the same operation.
    if (u.root != v.root)
      return false;
  }
  // Half-open interval overlap: [a, a+size) ∩ [b, b+size) ≠ ∅
  return u.byteOffset < v.byteOffset + v.byteSize &&
         v.byteOffset < u.byteOffset + u.byteSize;
}

} // namespace perf
