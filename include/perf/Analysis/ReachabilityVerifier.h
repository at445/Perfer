// ReachabilityVerifier.h - R1 reachability verifier.
//
// Spec §3 R1: every memref operand of every op must reside in a memory space
// reachable by the op's placement resource.
//
// Connectivity rules (hard-coded, per spec §1.3 op contracts):
//   - array  -> spad, acc
//   - dma_read  -> host (no space), spad, acc
//   - dma_write -> host (no space), spad, acc

#ifndef PERF_ANALYSIS_REACHABILITYVERIFIER_H
#define PERF_ANALYSIS_REACHABILITYVERIFIER_H

#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/LogicalResult.h"

namespace perf {

/// Verify R1 reachability for all perf ops in a module.
/// Emits diagnostics on failure.
llvm::LogicalResult verifyReachability(mlir::ModuleOp module);

} // namespace perf

#endif // PERF_ANALYSIS_REACHABILITYVERIFIER_H
