// WriteConflictVerifier.cpp - R4 write-conflict verifier.

#include "perf/Analysis/WriteConflictVerifier.h"

#include "perf/Analysis/BufferProvenance.h"
#include "perf/Dialect/PerfAttrs.h"
#include "perf/Dialect/PerfOps.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

#include <utility>
#include <vector>

using namespace mlir;

namespace perf {

namespace {

/// A single memory access performed by an executor op on one memref
/// operand.
struct Access {
  Operation *op;
  unsigned operandIndex;
  Provenance prov;
  bool isWrite;
};

/// Collect the token values an op depends on.
void collectTokenDeps(Operation *op, SmallVectorImpl<Value> &deps) {
  for (Value operand : op->getOperands()) {
    if (isa<TokenType>(operand.getType()))
      deps.push_back(operand);
  }
}

/// Compute the set of executor ops that must complete before `op` via the
/// token partial order.
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
      continue; // token.init or block arg: a source
    Operation *producer = it->second;
    if (ancestors.insert(producer).second)
      collectTokenDeps(producer, worklist);
  }
}

/// Gather the read/write direction of each memref operand of an executor
/// op from its MemoryEffects declaration.
void gatherAccesses(Operation *op,
                    const DenseMap<Value, Operation *> &tokenProducer,
                    std::vector<Access> &out) {
  auto effectOp = dyn_cast<MemoryEffectOpInterface>(op);
  if (!effectOp)
    return;

  SmallVector<SideEffects::EffectInstance<MemoryEffects::Effect>> effects;
  effectOp.getEffects(effects);

  DenseMap<unsigned, bool> dir; // operandIndex -> isWrite
  for (const auto &inst : effects) {
    Value v = inst.getValue();
    if (!v)
      continue;
    // Recover the operand index by matching the effect's value back to the
    // op's operand list.
    for (auto &opOperand : op->getOpOperands()) {
      if (opOperand.get() != v)
        continue;
      unsigned idx = opOperand.getOperandNumber();
      if (isa<MemoryEffects::Write>(inst.getEffect()))
        dir[idx] = true;
      else if (isa<MemoryEffects::Read>(inst.getEffect()) && !dir.count(idx))
        dir[idx] = false;
    }
  }

  for (auto &[idx, isWrite] : dir) {
    Value v = op->getOperand(idx);
    auto provOr = resolve(v);
    if (failed(provOr))
      continue; // statically unresolvable views are rejected elsewhere
    out.push_back(Access{op, idx, *provOr, isWrite});
  }
}

} // namespace

llvm::LogicalResult verifyWriteConflict(ModuleOp module) {
  bool failed = false;

  DenseMap<Value, Operation *> tokenProducer;
  module.walk([&](Operation *op) {
    if (isa<DmaLoadOp, DmaStoreOp, ArrayInitOp, ComputeOp>(op))
      tokenProducer[op->getResult(0)] = op;
  });

  std::vector<Access> accesses;
  module.walk([&](Operation *op) {
    if (isa<DmaLoadOp, DmaStoreOp, ArrayInitOp, ComputeOp>(op))
      gatherAccesses(op, tokenProducer, accesses);
  });

  // Transitive token ancestors per op, computed once.
  DenseMap<Operation *, DenseSet<Operation *>> ancestors;
  for (const auto &a : accesses) {
    if (!ancestors.count(a.op))
      tokenAncestors(a.op, tokenProducer, ancestors[a.op]);
  }

  auto ordered = [&](Operation *x, Operation *y) {
    return ancestors[y].contains(x) || ancestors[x].contains(y);
  };

  for (std::size_t i = 0; i < accesses.size(); ++i) {
    for (std::size_t j = i + 1; j < accesses.size(); ++j) {
      const Access &u = accesses[i];
      const Access &v = accesses[j];
      if (u.op == v.op)
        continue;
      if (!u.isWrite && !v.isWrite)
        continue; // read-read: performance dimension, not legality
      if (!conflict(u.prov, v.prov))
        continue;
      if (ordered(u.op, v.op))
        continue; // a happens-before edge already orders the pair

      failed = true;
      u.op->emitError()
          << "R4: write conflict between unordered accesses on the same "
             "buffer: this "
          << (u.isWrite ? "write" : "read") << " (operand " << u.operandIndex
          << ") conflicts with an unordered "
          << (v.isWrite ? "write" : "read") << " in '" << v.op->getName()
          << "'";
      v.op->emitRemark()
          << "R4: conflicting access (operand " << v.operandIndex
          << ") is not ordered against the other by the token partial order";
    }
  }

  return failed ? failure() : success();
}

} // namespace perf
