// ArrayStateVerifier.h - R6 array PE-state provenance verifier.
//
// Spec section 3 R6: the PE accumulator state every perf.compute
// accumulates onto must have a provenance — walking the token chain back
// from a compute must reach either a perf.array_init or another
// perf.compute. Accumulating onto register state of unknown origin is
// rejected.
//
// Additionally, an array_init (which overwrites the PE register state)
// must be ordered after the previous tile's writeback: the writeback must
// complete before the init begins (WAR), carried by the token partial
// order.

#ifndef PERF_ANALYSIS_ARRAYSTATEVERIFIER_H
#define PERF_ANALYSIS_ARRAYSTATEVERIFIER_H

#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/LogicalResult.h"

namespace perf {

/// Verify R6 PE-state provenance for all compute / array_init ops.
/// Emits diagnostics on failure.
llvm::LogicalResult verifyArrayState(mlir::ModuleOp module);

} // namespace perf

#endif // PERF_ANALYSIS_ARRAYSTATEVERIFIER_H
