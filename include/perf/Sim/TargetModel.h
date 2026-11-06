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

struct SramConfig {
  int64_t capacityKib;
  int64_t banks;
  int64_t readPorts;
  int64_t writePorts;
  int64_t portWidthBytes;

  int64_t capacityBytes() const;
};

struct ResourceInstance {
  std::string kind;
  int64_t index;
  bool inOrder;
  bool placeable;

  std::string name() const;  // "dma_read#0"
};

//===----------------------------------------------------------------------===//
// TargetModel
//===----------------------------------------------------------------------===//

class TargetModel {
public:
  static llvm::FailureOr<TargetModel> build(perf::TargetAttr attr);

  perf::TargetAttr getAttr() const { return attr_; }

  const DmaConfig &dma() const { return dma_; }
  const ArrayConfig &array() const { return array_; }
  const SramConfig &scratchpad() const { return scratchpad_; }
  const SramConfig &accumulator() const { return accumulator_; }

  llvm::ArrayRef<ResourceInstance> instances() const { return instances_; }

  const ResourceInstance *findInstance(llvm::StringRef kind,
                                        int64_t index) const;
  const ResourceInstance *findInstance(llvm::StringRef name) const;

private:
  explicit TargetModel(perf::TargetAttr attr) : attr_(attr) {}

  perf::TargetAttr attr_;
  DmaConfig dma_;
  ArrayConfig array_;
  SramConfig scratchpad_;
  SramConfig accumulator_;
  llvm::SmallVector<ResourceInstance> instances_;
};

} // namespace perf

#endif // PERF_SIM_TARGETMODEL_H
