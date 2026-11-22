// TargetModel.h - Strongly-typed hardware configuration for simulation.

#ifndef PERF_SIM_TARGETMODEL_H
#define PERF_SIM_TARGETMODEL_H

#include "perf/Dialect/PerfAttrs.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/LogicalResult.h"

#include <cstdint>
#include <optional>
#include <string>

namespace perf {

//===----------------------------------------------------------------------===//
// Config structs (snapshots, not live views)
//===----------------------------------------------------------------------===//

struct DmaConfig {
  int64_t busBytesPerCycle;
  int64_t fixedLatencyCycles;
  int64_t maxOutstanding;
  int64_t burstBytes;

  int64_t transferCycles(int64_t bytes) const;
  int64_t totalLatency(int64_t bytes) const;
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

  int64_t numArrayInstances() const;
};

struct MemSpaceConfig {
  std::string name;
  int64_t capacityKib;
  int64_t banks;
  int64_t readPorts;
  int64_t writePorts;
  int64_t portWidthBytes;
  int64_t rowWidthBytes;

  int64_t capacityBytes() const { return capacityKib * 1024; }
  int64_t rowsPerBank() const {
    return capacityBytes() / (banks * rowWidthBytes);
  }
};

struct ExecutorInstance {
  std::string kind;
  int64_t index;
  bool inOrder;
  int64_t familyIndex;  // index into the family's config entry

  std::string name() const;  // "dma_read#0"
};

struct ExecutorFamilyModel {
  std::string kind;
  int64_t count;
  bool inOrder;
  llvm::SmallVector<std::string> spaces;  // reachable memory spaces (R1)

  // Family-level microarchitectural config. Exactly one is populated,
  // according to the family's kind:
  //   dma_read / dma_write -> dma,  array -> arrayConfig.
  std::optional<DmaConfig> dma;
  std::optional<ArrayConfig> arrayConfig;

  bool canReach(llvm::StringRef space) const;
};

struct LinkInstance {
  std::string kind;
  int64_t index;

  std::string name() const;  // "dram_bus#0"
};

//===----------------------------------------------------------------------===//
// TargetModel
//===----------------------------------------------------------------------===//

class TargetModel {
public:
  static llvm::FailureOr<TargetModel> build(perf::TargetAttr attr);

  perf::TargetAttr getAttr() const { return attr_; }

  llvm::ArrayRef<MemSpaceConfig> memSpaces() const { return memSpaces_; }
  const MemSpaceConfig *findMemSpace(llvm::StringRef name) const;

  llvm::ArrayRef<ExecutorFamilyModel> families() const { return families_; }
  const ExecutorFamilyModel *findFamily(llvm::StringRef kind) const;

  llvm::ArrayRef<ExecutorInstance> executors() const { return executors_; }
  llvm::ArrayRef<LinkInstance> links() const { return links_; }

  const ExecutorInstance *findExecutor(llvm::StringRef kind,
                                       int64_t index) const;
  const ExecutorInstance *findExecutor(llvm::StringRef name) const;
  const LinkInstance *findLink(llvm::StringRef kind, int64_t index) const;

  // Family-level config lookup by executor kind (returns nullptr when the
  // kind is absent or carries a different config type).
  const DmaConfig *dmaConfig(llvm::StringRef kind) const;
  const ArrayConfig *arrayConfig(llvm::StringRef kind) const;

  // R1 connectivity: does executor kind reach the given memory space?
  bool executorCanReach(llvm::StringRef kind, llvm::StringRef space) const;

private:
  explicit TargetModel(perf::TargetAttr attr) : attr_(attr) {}

  perf::TargetAttr attr_;
  llvm::SmallVector<MemSpaceConfig> memSpaces_;
  llvm::SmallVector<ExecutorFamilyModel> families_;
  llvm::SmallVector<ExecutorInstance> executors_;
  llvm::SmallVector<LinkInstance> links_;
};

} // namespace perf

#endif // PERF_SIM_TARGETMODEL_H
