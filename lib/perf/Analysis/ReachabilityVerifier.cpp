// ReachabilityVerifier.cpp - R1 reachability verifier against the contract.

#include "perf/Analysis/ReachabilityVerifier.h"

#include "perf/Dialect/PerfAttrs.h"
#include "perf/Dialect/PerfOps.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"

using namespace mlir;

namespace perf {

namespace {

/// Verify one op's placement resource and memref operands against the
/// projected contract.
LogicalResult verifyOp(Operation *op, const TargetModel &target) {
  auto resAttr = op->getAttrOfType<ResRefAttr>("resource");
  if (!resAttr) {
    op->emitError() << "R1: op missing 'resource' attribute";
    return failure();
  }

  StringRef kind = resAttr.getKind();
  int64_t index = resAttr.getIndex();

  // The placement must reference an executor instance that exists in the
  // contract. Links are not executors, so they are rejected here by
  // construction.
  const ExecutorInstance *inst = target.findExecutor(kind, index);
  if (!target.findFamily(kind)) {
    return op->emitError() << "R1: unknown executor kind '" << kind << "'";
  }
  if (!inst) {
    return op->emitError() << "R1: executor '" << kind << "#" << index
                           << "' out of range";
  }

  const ExecutorFamilyModel *fam = target.findFamily(kind);

  for (Value operand : op->getOperands()) {
    auto memrefType = dyn_cast<MemRefType>(operand.getType());
    if (!memrefType)
      continue; // skip token operands

    auto spaceAttr =
        dyn_cast_if_present<SpaceAttr>(memrefType.getMemorySpace());

    if (!spaceAttr) {
      // Host-side operand: reachable iff the resource is a DMA bridge.
      if (!fam->dma) {
        return op->emitError()
               << "R1: executor '" << kind
               << "' cannot access host memory (not a DMA bridge)";
      }
      continue;
    }

    StringRef spaceName = spaceAttr.getName();
    if (!target.findMemSpace(spaceName)) {
      return op->emitError() << "R1: unknown memory space '" << spaceName
                             << "' (not declared in the design)";
    }
    if (!fam->canReach(spaceName)) {
      return op->emitError() << "R1: executor '" << kind
                             << "' cannot reach memory space '" << spaceName
                             << "'";
    }
  }

  return success();
}

} // namespace

llvm::LogicalResult verifyReachability(ModuleOp module,
                                       const TargetModel &target) {
  bool failed = false;
  module.walk([&](Operation *op) {
    if (isa<DmaLoadOp, DmaStoreOp, ArrayInitOp, ComputeOp>(op)) {
      if (::mlir::failed(verifyOp(op, target)))
        failed = true;
    }
  });
  return failed ? failure() : success();
}

} // namespace perf
