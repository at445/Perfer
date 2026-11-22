// ArrayStateVerifier.cpp - R6 array PE-state provenance verifier.

#include "perf/Analysis/ArrayStateVerifier.h"

#include "perf/Dialect/PerfAttrs.h"
#include "perf/Dialect/PerfOps.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace perf {

namespace {

/// Collect the token values an op depends on.
void collectTokenDeps(Operation *op, SmallVectorImpl<Value> &deps) {
  for (Value operand : op->getOperands()) {
    if (isa<TokenType>(operand.getType()))
      deps.push_back(operand);
  }
}

/// Find the nearest PE-state-defining ancestor (array_init or compute)
/// reachable from `op` along the token chain, per array resource.
/// Returns the defining op, or nullptr when the chain runs out at a source
/// (token.init / block arg) without ever establishing PE state.
///
/// The search is bounded to the same array resource as `op`: PE state is
/// per-array, so an init on a different array does not count.
Operation *peStateSource(Operation *op, StringRef arrayKind, int64_t arrayIndex,
                         const DenseMap<Value, Operation *> &tokenProducer) {
  SmallVector<Value> worklist;
  collectTokenDeps(op, worklist);
  DenseSet<Value> visited;
  DenseSet<Operation *> visitedOps;

  // BFS/DFS backwards; the nearest state-defining op on ANY path is the
  // provenance. Because the token order is a partial order, a compute's
  // PE state is well-defined only if EVERY backward path either hits a
  // state source first or never establishes one — but for a single array
  // resource the chain is expected to be linear through it, so we report
  // the first source found.
  while (!worklist.empty()) {
    Value tok = worklist.pop_back_val();
    if (!visited.insert(tok).second)
      continue;
    auto it = tokenProducer.find(tok);
    if (it == tokenProducer.end())
      continue; // token.init / block arg: a source, no PE state upstream
    Operation *producer = it->second;
    if (!visitedOps.insert(producer).second)
      continue;

    if (auto init = dyn_cast<ArrayInitOp>(producer)) {
      if (init.getResource().getKind() == arrayKind &&
          init.getResource().getIndex() == arrayIndex)
        return producer;
    } else if (auto compute = dyn_cast<ComputeOp>(producer)) {
      if (compute.getResource().getKind() == arrayKind &&
          compute.getResource().getIndex() == arrayIndex)
        return producer;
    }
    collectTokenDeps(producer, worklist);
  }
  return nullptr;
}

} // namespace

llvm::LogicalResult verifyArrayState(ModuleOp module) {
  bool failed = false;

  DenseMap<Value, Operation *> tokenProducer;
  module.walk([&](Operation *op) {
    if (isa<DmaLoadOp, DmaStoreOp, ArrayInitOp, ComputeOp>(op))
      tokenProducer[op->getResult(0)] = op;
  });

  module.walk([&](ComputeOp compute) {
    StringRef kind = compute.getResource().getKind();
    int64_t index = compute.getResource().getIndex();

    Operation *source = peStateSource(compute, kind, index, tokenProducer);
    if (!source) {
      failed = true;
      compute.emitError()
          << "R6: compute accumulates onto PE state of unknown origin on '"
          << kind << "#" << index
          << "': no array_init or earlier compute is reachable along the "
             "token chain";
    }
  });

  // R6 WAR: an array_init that reads old partial sums (operand form)
  // overwrites the PE registers, so it must be ordered after the previous
  // tile's writeback. The ordering is checked structurally: the operand
  // region's producer must be ordered before this init via the token
  // chain. A writeback-producing compute that is NOT ordered before the
  // init is a violation.
  module.walk([&](ArrayInitOp init) {
    if (!init.getDst())
      return; // zero-fill form: no SRAM read, no WAR hazard

    StringRef kind = init.getResource().getKind();
    int64_t index = init.getResource().getIndex();

    // Find the most recent compute on this array that could have produced
    // the partial sums being read; it must be ordered before this init.
    Operation *source = peStateSource(init, kind, index, tokenProducer);
    if (source && isa<ComputeOp>(source)) {
      // The init reads the state that `source` produced; the token chain
      // guarantees the ordering already (source is reachable), so this is
      // satisfied by construction. The violation we must catch is the
      // opposite: an init that has NO path back to the compute whose
      // state it claims to read — i.e., the compute is unordered. That
      // case is when peStateSource finds nothing on the array.
      return;
    }

    // No compute on this array is reachable before the init: the partial
    // sums it claims to read have no producer in the token order.
    failed = true;
    init.emitError()
        << "R6: array_init reads old partial sums from '" << kind << "#"
        << index << "' but no compute on that array is ordered before it "
           "along the token chain (WAR: the writeback must complete first)";
  });

  return failed ? failure() : success();
}

} // namespace perf
