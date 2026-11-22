// ReachabilityVerifier.h - R1 reachability verifier.
//
// Spec §3 R1: every memref operand of every op must reside in a memory
// space reachable by the op's placement resource. Bank assignment is
// derived from row addresses and plays no role in R1.
//
// The verifier consumes the projected compiler-simulator contract
// (TargetModel):
//   - the placement resource must exist (kind known, index in range);
//   - each on-chip operand's space must be reachable by that resource
//     (executor family's connectivity);
//   - host-side operands (no #perf.space) are reachable iff the resource
//     is a DMA bridge (its family carries a #perf.dma config).

#ifndef PERF_ANALYSIS_REACHABILITYVERIFIER_H
#define PERF_ANALYSIS_REACHABILITYVERIFIER_H

#include "perf/Sim/TargetModel.h"

#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/LogicalResult.h"

namespace perf {

/// Verify R1 reachability for all perf ops in a module against the
/// projected contract. Emits diagnostics on failure.
llvm::LogicalResult verifyReachability(mlir::ModuleOp module,
                                       const TargetModel &target);

} // namespace perf

#endif // PERF_ANALYSIS_REACHABILITYVERIFIER_H
