// ExecutionView.cpp

#include "perf/Sim/ExecutionView.h"

using namespace mlir;

namespace perf {

ExecutionView ExecutionView::build(ModuleOp module) {
  ExecutionView view;
  view.module_ = module;

  module.walk([&](Operation *op) {
    if (!isa<DmaLoadOp, DmaStoreOp, ArrayInitOp, ComputeOp>(op))
      return;

    OpId id = static_cast<OpId>(view.ops_.size());
    view.ops_.push_back(op);
    view.opToId_[op] = id;

    auto resAttr = op->getAttrOfType<ResRefAttr>("resource");
    view.placement_[op] = {resAttr.getKind().str(), resAttr.getIndex()};

    view.tokenProducer_[op->getResult(0)] = op;
  });

  return view;
}

} // namespace perf
