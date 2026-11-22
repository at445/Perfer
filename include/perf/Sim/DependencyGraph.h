// DependencyGraph.h - Token use-def dependency graph over executor ops.
//
// Architecture doc section 4.4: cross-resource waiting can only come from
// the dependency graph. Without it the simulator would pack every resource
// back to back and systematically erase cross-resource bubbles.
//
// Built from the token Value use-def edges cached in ExecutionView. Note
// the spec's division of labor: token use-def resolves ordering only; the
// legality of mutable-buffer RAW/WAR/WAW is R4's job in the verifier, the
// two do not overlap.

#ifndef PERF_SIM_DEPENDENCYGRAPH_H
#define PERF_SIM_DEPENDENCYGRAPH_H

#include "perf/Sim/ExecutionView.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>

namespace perf {

class DependencyGraph {
public:
  /// Build predecessor/successor lists from the view's token use-def edges.
  static DependencyGraph build(const ExecutionView &view);

  llvm::ArrayRef<OpId> predecessors(OpId id) const { return preds_[id]; }
  llvm::ArrayRef<OpId> successors(OpId id) const { return succs_[id]; }

  /// All predecessors are done (a source op is trivially ready).
  bool readyByDeps(OpId id) const {
    for (OpId p : preds_[id]) {
      if (!done_.contains(p))
        return false;
    }
    return true;
  }

  void markDone(OpId id, std::uint64_t doneTime) {
    done_.insert(id);
    doneTime_[id] = doneTime;
  }

  bool isDone(OpId id) const { return done_.contains(id); }

  std::uint64_t doneTime(OpId id) const {
    auto it = doneTime_.find(id);
    return it == doneTime_.end() ? 0 : it->second;
  }

private:
  llvm::SmallVector<llvm::SmallVector<OpId>> preds_;
  llvm::SmallVector<llvm::SmallVector<OpId>> succs_;
  llvm::DenseSet<OpId> done_;
  llvm::DenseMap<OpId, std::uint64_t> doneTime_;
};

} // namespace perf

#endif // PERF_SIM_DEPENDENCYGRAPH_H
