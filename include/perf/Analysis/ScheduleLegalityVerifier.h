// ScheduleLegalityVerifier.h - R3 schedule legality verifier.
//
// Spec section 3 R3: the total order of every in-order resource (the
// default block program order, or the order given by `perf.order`
// annotations) must be a linear extension of the token partial order.
// A violation means replay deadlock: the head of the resource queue
// waits for a predecessor that is not the head and can never issue.
//
// Default program order is always consistent (SSA requires definition
// before use), so R3 only has teeth when `perf.order` annotations give
// a non-default order.

#ifndef PERF_ANALYSIS_SCHEDULELEGALITYVERIFIER_H
#define PERF_ANALYSIS_SCHEDULELEGALITYVERIFIER_H

#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/LogicalResult.h"

namespace perf {

/// Verify R3 schedule legality for all executor ops in a module.
/// Emits diagnostics on failure.
llvm::LogicalResult verifyScheduleLegality(mlir::ModuleOp module);

} // namespace perf

#endif // PERF_ANALYSIS_SCHEDULELEGALITYVERIFIER_H
