// WriteConflictVerifier.h - R4 write-conflict verifier.
//
// Spec section 3 R4: two accesses that conflict (same root buffer with
// overlapping byte ranges, per BufferProvenance), at least one of which is
// a write, and that are unordered under the token partial order, are a
// write conflict and are rejected.
//
// Read-read conflicts are not a legality issue: their order is a pure
// performance dimension left to the schedule search space (spec section 3,
// R3 note).

#ifndef PERF_ANALYSIS_WRITECONFLICTVERIFIER_H
#define PERF_ANALYSIS_WRITECONFLICTVERIFIER_H

#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/LogicalResult.h"

namespace perf {

/// Verify R4 write-conflict freedom for all executor ops in a module.
/// Emits diagnostics on failure.
llvm::LogicalResult verifyWriteConflict(mlir::ModuleOp module);

} // namespace perf

#endif // PERF_ANALYSIS_WRITECONFLICTVERIFIER_H
