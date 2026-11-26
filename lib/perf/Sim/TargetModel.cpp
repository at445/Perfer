// TargetModel.cpp

#include "perf/Sim/TargetModel.h"

#include "perf/Dialect/PerfOps.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace perf;

//===----------------------------------------------------------------------===//
// DmaConfig
//===----------------------------------------------------------------------===//

int64_t DmaConfig::transferCycles(int64_t bytes,
                                  int64_t bytesPerCycle) const {
  return divideCeil(bytes, bytesPerCycle);
}

int64_t DmaConfig::totalLatency(int64_t bytes, int64_t bytesPerCycle) const {
  return fixedLatencyCycles + transferCycles(bytes, bytesPerCycle);
}

//===----------------------------------------------------------------------===//
// ArrayConfig
//===----------------------------------------------------------------------===//

int64_t ArrayConfig::numArrayInstances() const {
  return tileRows * tileCols;
}

//===----------------------------------------------------------------------===//
// ExecutorInstance / LinkInstance
//===----------------------------------------------------------------------===//

std::string ExecutorInstance::name() const {
  return kind + "#" + std::to_string(index);
}

bool ExecutorInstance::canReach(StringRef space) const {
  return llvm::is_contained(spaces, space);
}

std::string LinkInstance::name() const {
  return kind + "#" + std::to_string(index);
}

//===----------------------------------------------------------------------===//
// TargetModel
//===----------------------------------------------------------------------===//

namespace {

DmaConfig toDmaConfig(perf::DmaAttr a) {
  return {
      .readPorts = a.getReadPorts(),
      .writePorts = a.getWritePorts(),
      .fixedLatencyCycles = a.getFixedLatencyCycles(),
      .maxOutstanding = a.getMaxOutstanding(),
      .burstBytes = a.getBurstBytes(),
  };
}

ArrayConfig toArrayConfig(perf::ArrayAttr a) {
  return {
      .topology = a.getTopology().str(),
      .meshRows = a.getMeshRows(),
      .meshCols = a.getMeshCols(),
      .tileRows = a.getTileRows(),
      .tileCols = a.getTileCols(),
      .peLatencyCycles = a.getPeLatencyCycles(),
      .inputSkew = a.getInputSkew(),
      .interTilePipelined = a.getInterTilePipelined(),
      .accReadScaleCycles = a.getAccReadScaleCycles(),
      .accReadActCycles = a.getAccReadActCycles(),
  };
}

} // namespace

FailureOr<TargetModel> TargetModel::build(perf::TargetAttr target,
                                          perf::PriorAttr prior) {
  TargetModel model(target, prior);

  // ── memory spaces ────────────────────────────────────────────────────
  // The IDs that program-side #perf.space<name, ...> attributes resolve
  // against. The `banks` / `interleave` geometry is kept as-is; the
  // simulator derives per-bank contention ports from it at demand time.
  llvm::SmallDenseSet<llvm::StringRef> seenSpaces;
  for (mlir::Attribute a : target.getMemorySpaces()) {
    auto ms = cast<perf::MemSpaceAttr>(a);
    if (!seenSpaces.insert(ms.getName()).second) {
      llvm::errs() << "error: duplicate memory space name '" << ms.getName()
                   << "' in target '" << target.getTargetName() << "'\n";
      return failure();
    }
    model.memSpaces_.push_back({
        .name = ms.getName().str(),
        .capacityKib = ms.getCapacityKib(),
        .banks = ms.getBanks(),
        .readPorts = ms.getReadPorts(),
        .writePorts = ms.getWritePorts(),
        .portWidthBytes = ms.getPortWidthBytes(),
        .rowWidthBytes = ms.getRowWidthBytes(),
        .interleave = ms.getInterleave().str(),
    });
  }

  // ── links ────────────────────────────────────────────────────────────
  // dram_bus channels are the bandwidth source for DMA transfer nominal;
  // their per-channel bandwidth is looked up directly from the link list.
  for (mlir::Attribute a : target.getLinks()) {
    auto link = cast<perf::LinkAttr>(a);
    for (int64_t i = 0; i < link.getCount(); ++i) {
      model.links_.push_back({
          .kind = link.getKind().str(),
          .index = i,
          .bytesPerCycle = link.getBytesPerCycle(),
      });
    }
  }

  // ── executors ────────────────────────────────────────────────────────
  // Each executor family expands into `count` instances, each carrying its
  // own kind / index / ordering / reachable spaces and engine config.
  for (mlir::Attribute a : target.getExecutors()) {
    auto famAttr = cast<perf::ExecutorFamilyAttr>(a);

    std::optional<DmaConfig> dma;
    std::optional<ArrayConfig> array;
    mlir::Attribute config = famAttr.getConfig();
    if (auto d = dyn_cast<perf::DmaAttr>(config))
      dma = toDmaConfig(d);
    else if (auto ar = dyn_cast<perf::ArrayAttr>(config))
      array = toArrayConfig(ar);
    else {
      llvm::errs() << "error: executor kind '" << famAttr.getKind()
                   << "' carries an unsupported config type in target '"
                   << target.getTargetName() << "'\n";
      return failure();
    }

    llvm::SmallVector<std::string> spaces;
    for (mlir::Attribute s : famAttr.getSpaces())
      spaces.push_back(cast<mlir::StringAttr>(s).getValue().str());

    for (int64_t i = 0; i < famAttr.getCount(); ++i) {
      ExecutorInstance inst;
      inst.kind = famAttr.getKind().str();
      inst.index = i;
      inst.spaces = spaces;
      inst.dma = dma;
      inst.arrayConfig = array;
      model.executors_.push_back(std::move(inst));
    }
  }

  return model;
}

const SRAMConfig *TargetModel::findMemSpace(StringRef name) const {
  for (const auto &ms : memSpaces_) {
    if (ms.name == name)
      return &ms;
  }
  return nullptr;
}

const ExecutorInstance *TargetModel::findExecutor(StringRef kind,
                                                  int64_t index) const {
  for (const auto &inst : executors_) {
    if (inst.kind == kind && inst.index == index)
      return &inst;
  }
  return nullptr;
}

const ExecutorInstance *TargetModel::findExecutor(StringRef name) const {
  for (const auto &inst : executors_) {
    if (inst.name() == name)
      return &inst;
  }
  return nullptr;
}

const LinkInstance *TargetModel::findLink(StringRef kind, int64_t index) const {
  for (const auto &inst : links_) {
    if (inst.kind == kind && inst.index == index)
      return &inst;
  }
  return nullptr;
}

const DmaConfig *TargetModel::dmaConfig(StringRef kind) const {
  for (const auto &inst : executors_) {
    if (inst.kind == kind && inst.dma)
      return &*inst.dma;
  }
  return nullptr;
}

const ArrayConfig *TargetModel::arrayConfig(StringRef kind) const {
  for (const auto &inst : executors_) {
    if (inst.kind == kind && inst.arrayConfig)
      return &*inst.arrayConfig;
  }
  return nullptr;
}

int64_t TargetModel::dramBytesPerCycle() const {
  const auto *link = findLink("dram_bus", 0);
  return link ? link->bytesPerCycle : 0;
}

bool TargetModel::executorCanReach(StringRef kind, StringRef space) const {
  for (const auto &inst : executors_) {
    if (inst.kind == kind)
      return inst.canReach(space);
  }
  return false;
}
