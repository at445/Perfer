// ExecutionView.h - Lightweight index over a verified MLIR module.
//
// Architecture doc section 4.2: MLIR already provides Operation, Value,
// use-def, attributes and locations — there is no need to deep-copy a
// StaticOp/ProgramModel. ExecutionView is only an index:
//
//   - a stable integer OpId per executor op (event ordering, reports);
//   - cached resource placement (avoid re-parsing the resource attr in
//     the event loop);
//   - cached token producers (speeds up DependencyGraph construction).
//
// Lifetime: the module must be immutable for the view's lifetime; rebuild
// the view after any pass that rewrites the module.

#ifndef PERF_SIM_EXECUTIONVIEW_H
#define PERF_SIM_EXECUTIONVIEW_H

#include "perf/Dialect/PerfOps.h"

#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <string>
#include <utility>

namespace perf {

/// Stable integer identity for an executor op (event ordering, reports).
using OpId = std::uint32_t;

/// A resource instance reference, e.g. ("dma_read", 0) == dma_read#0.
using ResourceId = std::pair<std::string, int64_t>;

class ExecutionView {
public:
  /// Build the index over a verified module. Executor ops (dma_load,
  /// dma_store, array_init, compute) are indexed in block program order.
  static ExecutionView build(mlir::ModuleOp module);

  mlir::ModuleOp module() const { return module_; }

  llvm::ArrayRef<mlir::Operation *> ops() const { return ops_; }

  /// Look up the OpId of an executor op (nullptr -> invalid sentinel).
  OpId idOf(mlir::Operation *op) const {
    auto it = opToId_.find(op);
    return it == opToId_.end() ? kInvalidOpId : it->second;
  }

  mlir::Operation *opOf(OpId id) const { return ops_[id]; }

  /// Resource placement of an executor op, parsed from its resource attr.
  ResourceId placementOf(mlir::Operation *op) const {
    return placement_.find(op)->second;
  }

  /// token Value -> producing op (executor ops only; token.init and block
  /// arguments are not producers).
  const llvm::DenseMap<mlir::Value, mlir::Operation *> &tokenProducer() const {
    return tokenProducer_;
  }

  static constexpr OpId kInvalidOpId = ~OpId(0);

private:
  mlir::ModuleOp module_;
  llvm::SmallVector<mlir::Operation *> ops_;        // block program order
  llvm::DenseMap<mlir::Operation *, OpId> opToId_;
  llvm::DenseMap<mlir::Operation *, ResourceId> placement_;
  llvm::DenseMap<mlir::Value, mlir::Operation *> tokenProducer_;
};

} // namespace perf

#endif // PERF_SIM_EXECUTIONVIEW_H
