// ReachabilityVerifier.cpp - R1 reachability verifier.

#include "perf/Analysis/ReachabilityVerifier.h"

#include "perf/Dialect/PerfAttrs.h"
#include "perf/Dialect/PerfOps.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"

using namespace mlir;

namespace perf {

namespace {

/// Check if a resource kind can access a memory space.
/// host = no #perf.space attribute (DRAM side).
bool canAccess(StringRef resourceKind, bool isHost, StringRef spaceName) {
  if (resourceKind == "array")
    return !isHost && (spaceName == "spad" || spaceName == "acc");
  if (resourceKind == "dma_read" || resourceKind == "dma_write")
    return true; // DMA bridges host and on-chip
  return false;  // unknown resource kind
}

/// Verify one op's memref operands against its placement resource.
LogicalResult verifyOp(Operation *op) {
  auto resAttr = op->getAttrOfType<ResRefAttr>("resource");
  if (!resAttr) {
    op->emitError() << "R1: op missing 'resource' attribute";
    return failure();
  }

  StringRef resourceKind = resAttr.getKind();

  for (Value operand : op->getOperands()) {
    auto memrefType = dyn_cast<MemRefType>(operand.getType());
    if (!memrefType)
      continue; // skip token operands

    auto spaceAttr = dyn_cast_if_present<SpaceAttr>(
        memrefType.getMemorySpace());
    bool isHost = !spaceAttr;
    StringRef spaceName = spaceAttr ? spaceAttr.getName() : "";

    if (!canAccess(resourceKind, isHost, spaceName)) {
      op->emitError() << "R1: resource '" << resourceKind
                      << "' cannot access operand in "
                      << (isHost ? "host DRAM" : "space '" + spaceName.str() + "'");
      return failure();
    }
  }

  return success();
}

} // namespace

llvm::LogicalResult verifyReachability(ModuleOp module) {
  bool failed = false;
  module.walk([&](Operation *op) {
    if (isa<DmaLoadOp, DmaStoreOp, PreloadOp, ComputeOp>(op)) {
      if (::mlir::failed(verifyOp(op)))
        failed = true;
    }
  });
  return failed ? failure() : success();
}

} // namespace perf
