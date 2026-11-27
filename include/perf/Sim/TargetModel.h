// TargetModel.h - Strongly-typed hardware configuration for simulation.

#ifndef PERF_SIM_TARGETMODEL_H
#define PERF_SIM_TARGETMODEL_H

#include "perf/Dialect/PerfAttrs.h"
#include "perf/Support/OptRef.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/LogicalResult.h"

#include <cstdint>
#include <optional>
#include <string>

namespace perf {

//===----------------------------------------------------------------------===//
// Config structs (snapshots, not live views)
//===----------------------------------------------------------------------===//

// DMA engine-intrinsic parameters plus its read/write port counts. The
// transfer bandwidth is NOT here: it belongs to the shared DRAM channel
// (DramConfig / the dram_bus link), the source of transfer nominal.
struct DmaConfig {
  int64_t readPorts;
  int64_t writePorts;
  int64_t fixedLatencyCycles;
  int64_t maxOutstanding;
  int64_t burstBytes;

  // Transfer cycles need the channel bandwidth, supplied by the caller from
  // the dram link the engine's ports share.
  int64_t transferCycles(int64_t bytes, int64_t bytesPerCycle) const;
  int64_t totalLatency(int64_t bytes, int64_t bytesPerCycle) const;
};

struct ArrayConfig {
  std::string topology;
  int64_t meshRows;
  int64_t meshCols;
  int64_t tileRows;
  int64_t tileCols;
  int64_t peLatencyCycles;
  bool inputSkew;
  bool interTilePipelined;
  // Gemmini acc read-out epilogue (scale + activation) pipeline stages.
  int64_t accReadScaleCycles;
  int64_t accReadActCycles;

  int64_t numArrayInstances() const;
};

// The geometry and row-to-bank mapping of one on-chip SRAM space. This is a
// parameter source: R1 bounds checks use rowsPerBank/totalRows, and the
// simulator's demand generation uses banks/interleave to map a #perf.space
// access onto per-bank contention ports. Per-bank ports are derived by the
// simulator from `banks`; they are not materialized as model objects.
struct SRAMConfig {
  std::string name;
  int64_t capacityKib;
  int64_t banks;
  int64_t readPorts;
  int64_t writePorts;
  int64_t portWidthBytes;
  int64_t rowWidthBytes;
  // Row-to-bank interleave policy: "low" (bank = row % banks) or "high"
  // (bank = row / rowsPerBank). Used by the simulator's demand generation.
  std::string interleave;

  int64_t capacityBytes() const { return capacityKib * 1024; }
  int64_t rowsPerBank() const {
    return capacityBytes() / (banks * rowWidthBytes);
  }
  int64_t totalRows() const { return banks * rowsPerBank(); }
};

struct ExecutorInstance {
  std::string kind;
  int64_t index;
  llvm::SmallVector<std::string> spaces;  // reachable memory spaces (R1)

  // Microarchitectural config. Exactly one is populated, according to the
  // instance's kind:
  //   dma_read / dma_write -> dma,  array -> arrayConfig.
  std::optional<DmaConfig> dma;
  std::optional<ArrayConfig> arrayConfig;

  std::string name() const;  // "dma_read#0"

  bool canReach(llvm::StringRef space) const;
};

struct LinkInstance {
  std::string kind;
  int64_t index;
  int64_t bytesPerCycle;  // per-channel transfer bandwidth

  std::string name() const;  // "dram_bus#0"
};

//===----------------------------------------------------------------------===//
// TargetModel
//===----------------------------------------------------------------------===//

// Strongly-typed snapshot of a #perf.target, consumed one-way by the
// simulator and the verifier rules. The model does not write MLIR back: a
// malformed target surfaces through the verifier rules (R1, ...), not through
// any re-serialization here.
class TargetModel {
public:
  // Build from the simulator-level target attribute (#perf.target).
  static llvm::FailureOr<TargetModel> build(perf::TargetAttr target,
                                            perf::PriorAttr prior);

  perf::PriorAttr getPriorAttr() const { return priorAttr_; }

  llvm::ArrayRef<SRAMConfig> memSpaces() const { return memSpaces_; }
  OptCRef<SRAMConfig> findMemSpace(llvm::StringRef name) const;

  llvm::ArrayRef<ExecutorInstance> executors() const { return executors_; }
  llvm::ArrayRef<LinkInstance> links() const { return links_; }

  OptCRef<ExecutorInstance> findExecutor(llvm::StringRef kind,
                                         int64_t index) const;
  OptCRef<ExecutorInstance> findExecutor(llvm::StringRef name) const;
  OptCRef<LinkInstance> findLink(llvm::StringRef kind, int64_t index) const;

  // Config lookup by executor kind (returns std::nullopt when the kind is
  // absent or carries a different config type). Finds the first instance of
  // kind.
  OptCRef<DmaConfig> dmaConfig(llvm::StringRef kind) const;
  OptCRef<ArrayConfig> arrayConfig(llvm::StringRef kind) const;

  // Per-channel bandwidth of the shared dram_bus channel the DMA ports
  // contend on (the source of DMA transfer nominal). 0 when absent.
  int64_t dramBytesPerCycle() const;

  // R1 connectivity: does executor kind reach the given memory space?
  bool executorCanReach(llvm::StringRef kind, llvm::StringRef space) const;

private:
  explicit TargetModel(perf::TargetAttr target, perf::PriorAttr prior)
      : targetAttr_(target), priorAttr_(prior) {}

  perf::TargetAttr targetAttr_;
  perf::PriorAttr priorAttr_;
  llvm::SmallVector<SRAMConfig> memSpaces_;
  llvm::SmallVector<ExecutorInstance> executors_;
  llvm::SmallVector<LinkInstance> links_;
};

} // namespace perf

#endif // PERF_SIM_TARGETMODEL_H
