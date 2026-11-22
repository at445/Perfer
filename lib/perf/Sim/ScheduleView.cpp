// ScheduleView.cpp

#include "perf/Sim/ScheduleView.h"

#include "llvm/ADT/STLExtras.h"

using namespace mlir;

namespace perf {

ScheduleView ScheduleView::build(const ExecutionView &view) {
  ScheduleView sched;

  // Group ops per resource, preserving block program order (the view's
  // ops() are already in program order).
  for (Operation *op : view.ops()) {
    ResourceId res = view.placementOf(op);
    sched.queues_[res].push_back(view.idOf(op));
  }

  // Honor explicit per-op `order` annotations when present on every op of
  // a resource: sort stably by (order, program position). Mixed or absent
  // annotations keep the default program order.
  for (auto &[res, queue] : sched.queues_) {
    bool allAnnotated = llvm::all_of(queue, [&](OpId id) {
      return view.opOf(id)->hasAttr("order");
    });
    if (allAnnotated) {
      llvm::stable_sort(queue, [&](OpId a, OpId b) {
        int64_t oa = view.opOf(a)->getAttrOfType<IntegerAttr>("order").getInt();
        int64_t ob = view.opOf(b)->getAttrOfType<IntegerAttr>("order").getInt();
        if (oa != ob)
          return oa < ob;
        return a < b; // program-order tiebreak
      });
    }
    sched.headPtr_[res] = 0;
  }

  return sched;
}

OpId ScheduleView::head(const ResourceId &resource) const {
  auto qIt = queues_.find(resource);
  if (qIt == queues_.end())
    return ExecutionView::kInvalidOpId;
  auto pIt = headPtr_.find(resource);
  if (pIt == headPtr_.end() || pIt->second >= qIt->second.size())
    return ExecutionView::kInvalidOpId;
  return qIt->second[pIt->second];
}

void ScheduleView::advance(const ResourceId &resource) {
  auto pIt = headPtr_.find(resource);
  if (pIt != headPtr_.end())
    ++pIt->second;
}

} // namespace perf
