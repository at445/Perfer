// BufferProvenance.h - Static buffer identity and alias analysis.
//
// Maps each memref SSA value to its root buffer (perf.alloc or func block
// argument) plus a static byte range within that root. Used by the R1-R5
// verifier for conflict detection.

#ifndef PERF_ANALYSIS_BUFFERPROVENANCE_H
#define PERF_ANALYSIS_BUFFERPROVENANCE_H

#include "mlir/IR/Value.h"
#include "llvm/Support/LogicalResult.h"

#include <cstdint>

namespace mlir {
class Operation;
}

namespace perf {

/// Static provenance record for one memref value.
struct Provenance {
  /// Root operation: perf.alloc, or the parent func for block arguments.
  /// Null when the root is a block argument (see argIndex).
  mlir::Operation *root;
  /// For block-argument roots: the argument index. -1 for alloc roots.
  int argIndex;
  /// Byte offset relative to the root buffer's start.
  int64_t byteOffset;
  /// Byte size of the accessed range.
  int64_t byteSize;
  /// True if the root is a func block argument (host-side DRAM).
  bool isHost;
};

/// Resolve the static provenance of a memref value by walking its def-chain.
/// Fails when the chain cannot be statically resolved (dynamic offsets,
/// unsupported view ops, phi nodes, etc.).
llvm::FailureOr<Provenance> resolve(mlir::Value v);

/// Conflict test: same root and overlapping byte ranges.
bool conflict(const Provenance &u, const Provenance &v);

} // namespace perf

#endif // PERF_ANALYSIS_BUFFERPROVENANCE_H
