// ScheduleView.h - Per-resource playback order for the replay engine.
//
// Architecture doc section 4.5: the simulator replays the compiler's order
// and introduces no dynamic scheduling of its own. MVP uses the spec's
// default rule (section 4.1): ops on the same in-order resource replay in
// block program order. Explicit perf.schedule annotations are honored when
// present (R3 already guarantees they are a linear extension of the token
// partial order).

#ifndef PERF_SIM_SCHEDULEVIEW_H
#define PERF_SIM_SCHEDULEVIEW_H

#include "perf/Sim/ExecutionView.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

#include <map>
#include <utility>
#include <vector>

namespace perf {

class ScheduleView {
public:
  /// Build per-resource playback queues from the view. Ops are ordered by
  /// an optional `order` attribute when present on every op of a resource,
  /// otherwise by block program order (the spec default).
  static ScheduleView build(const ExecutionView &view);

  /// The op at the head of a resource's queue (kInvalidOpId when empty).
  OpId head(const ResourceId &resource) const;

  /// Pop the head of a resource's queue.
  void advance(const ResourceId &resource);

  bool isExhausted(const ResourceId &resource) const {
    return head(resource) == ExecutionView::kInvalidOpId;
  }

private:
  std::map<ResourceId, std::vector<OpId>> queues_;
  std::map<ResourceId, std::size_t> headPtr_;
};

} // namespace perf

#endif // PERF_SIM_SCHEDULEVIEW_H
