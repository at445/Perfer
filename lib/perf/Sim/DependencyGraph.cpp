// DependencyGraph.cpp

#include "perf/Sim/DependencyGraph.h"

using namespace mlir;

namespace perf {

DependencyGraph DependencyGraph::build(const ExecutionView &view) {
  DependencyGraph graph;
  const auto n = view.ops().size();
  graph.preds_.resize(n);
  graph.succs_.resize(n);

  const auto &tokenProducer = view.tokenProducer();
  for (Operation *op : view.ops()) {
    OpId consumerId = view.idOf(op);
    for (Value operand : op->getOperands()) {
      // Ordering is carried only by tokens (spec 1.3): mutable-memref
      // RAW/WAR/WAW must be tokenized by the compiler, so a memref SSA
      // use-def edge never adds an ordering constraint here.
      if (!isa<TokenType>(operand.getType()))
        continue;
      auto it = tokenProducer.find(operand);
      if (it == tokenProducer.end())
        continue; // token.init / block arg: a source, no edge
      OpId producerId = view.idOf(it->second);
      graph.preds_[consumerId].push_back(producerId);
      graph.succs_[producerId].push_back(consumerId);
    }
  }

  return graph;
}

} // namespace perf
