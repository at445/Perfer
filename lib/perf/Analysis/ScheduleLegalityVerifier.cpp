// ScheduleLegalityVerifier.cpp - R3 schedule legality verifier.

#include "perf/Analysis/ScheduleLegalityVerifier.h"

#include "perf/Dialect/PerfAttrs.h"
#include "perf/Dialect/PerfOps.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"

#include <map>
#include <string>
#include <utility>
#include <vector>

using namespace mlir;

namespace perf {

namespace {

/// Collect the token values an op depends on (its dependency operands).
void collectTokenDeps(Operation *op, SmallVectorImpl<Value> &deps) {
  for (Value operand : op->getOperands()) {
    if (isa<TokenType>(operand.getType()))
      deps.push_back(operand);
  }
}

/// Compute the set of executor ops that must complete before `op` via the
/// token partial order (transitive closure of token use-def edges).
void tokenAncestors(Operation *op,
                    const DenseMap<Value, Operation *> &tokenProducer,
                    DenseSet<Operation *> &ancestors) {
  SmallVector<Value> worklist;
  collectTokenDeps(op, worklist);
  DenseSet<Value> visited;
  while (!worklist.empty()) {
    Value tok = worklist.pop_back_val();
    if (!visited.insert(tok).second)
      continue;
    auto it = tokenProducer.find(tok);
    if (it == tokenProducer.end())
      continue; // token from token.init or block arg: a source
    Operation *producer = it->second;
    if (ancestors.insert(producer).second)
      collectTokenDeps(producer, worklist);
  }
}

} // namespace

llvm::LogicalResult verifyScheduleLegality(ModuleOp module) {
  bool failed = false;

  // Token producers and sched.name -> op, in block program order.
  DenseMap<Value, Operation *> tokenProducer;
  llvm::StringMap<Operation *> bySchedName;
  llvm::StringMap<int64_t> programPointOf;
  int64_t programPoint = 0;

  module.walk([&](Operation *op) {
    if (isa<DmaLoadOp, DmaStoreOp, ArrayInitOp, ComputeOp>(op)) {
      tokenProducer[op->getResult(0)] = op;
      if (auto nameAttr = op->getAttrOfType<StringAttr>("sched_name")) {
        StringRef name = nameAttr.getValue();
        if (bySchedName.contains(name)) {
          failed = true;
          op->emitError() << "R3: duplicate sched.name '" << name << "'";
          bySchedName[name]->emitRemark()
              << "R3: previous definition of sched.name '" << name << "'";
        }
        bySchedName[name] = op;
        programPointOf[name] = programPoint;
      }
    }
    ++programPoint;
  });

  // Transitive token ancestors for each named op, computed once.
  DenseMap<Operation *, DenseSet<Operation *>> ancestors;
  for (auto &[name, op] : bySchedName)
    tokenAncestors(op, tokenProducer, ancestors[op]);

  // Check every schedule entry: the name list is a total order on one
  // resource instance and must be a linear extension of the token
  // partial order.
  module.walk([&](ScheduleOp sched) {
    for (Attribute entryAttr : sched.getEntries()) {
      auto entry = cast<ScheduleEntryAttr>(entryAttr);
      ResRefAttr resource = entry.getResource();

      // Resolve the name list to ops, in declared execution order.
      std::vector<std::pair<StringRef, Operation *>> ordered;
      for (Attribute nameAttr : entry.getOps()) {
        StringRef name = cast<StringAttr>(nameAttr).getValue();
        auto it = bySchedName.find(name);
        if (it == bySchedName.end()) {
          failed = true;
          sched.emitError()
              << "R3: schedule entry for resource '"
              << resource.getKind() << "#" << resource.getIndex()
              << "' references unknown sched.name '" << name << "'";
          continue;
        }
        Operation *op = it->second;

        // The referenced op must actually be placed on this resource.
        auto resAttr = op->getAttrOfType<ResRefAttr>("resource");
        if (resAttr.getKind() != resource.getKind() ||
            resAttr.getIndex() != resource.getIndex()) {
          failed = true;
          sched.emitError()
              << "R3: schedule entry for resource '" << resource.getKind()
              << "#" << resource.getIndex() << "' references sched.name '"
              << name << "' but that op is placed on '"
              << resAttr.getKind() << "#" << resAttr.getIndex() << "'";
          continue;
        }
        ordered.emplace_back(name, op);
      }

      // For every pair (earlier, later) in the declared order, the token
      // partial order must not require later -> earlier.
      for (std::size_t i = 0; i < ordered.size(); ++i) {
        for (std::size_t j = i + 1; j < ordered.size(); ++j) {
          Operation *earlier = ordered[i].second;
          Operation *later = ordered[j].second;
          if (ancestors[earlier].contains(later)) {
            failed = true;
            sched.emitError()
                << "R3: schedule order on resource '" << resource.getKind()
                << "#" << resource.getIndex()
                << "' conflicts with token dependency: sched.name '"
                << ordered[i].first << "' is ordered before '"
                << ordered[j].first << "' but must complete after it";
            later->emitRemark()
                << "R3: token dependency requires this op to complete first";
          }
        }
      }
    }
  });

  return failed ? failure() : success();
}

} // namespace perf
