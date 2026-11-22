// TargetModel.cpp

#include "perf/Sim/TargetModel.h"

#include "mlir/IR/Attributes.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace perf;

//===----------------------------------------------------------------------===//
// DmaConfig
//===----------------------------------------------------------------------===//

int64_t DmaConfig::transferCycles(int64_t bytes) const {
  return divideCeil(bytes, busBytesPerCycle);
}

int64_t DmaConfig::totalLatency(int64_t bytes) const {
  return fixedLatencyCycles + transferCycles(bytes);
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

bool ExecutorFamilyModel::canReach(StringRef space) const {
  return llvm::is_contained(spaces, space);
}

std::string LinkInstance::name() const {
  return kind + "#" + std::to_string(index);
}

//===----------------------------------------------------------------------===//
// TargetModel
//===----------------------------------------------------------------------===//

FailureOr<TargetModel> TargetModel::build(perf::TargetAttr attr) {
  TargetModel model(attr);

  // Named memory spaces (schema v4). Names must be unique: they are the IDs
  // that program-side #perf.space<name, ...> attributes resolve against.
  llvm::SmallDenseSet<llvm::StringRef> seenSpaces;
  for (mlir::Attribute a : attr.getMemorySpaces()) {
    auto ms = cast<perf::MemSpaceAttr>(a);
    if (!seenSpaces.insert(ms.getName()).second) {
      llvm::errs() << "error: duplicate memory space name '" << ms.getName()
                   << "' in target '" << attr.getTargetName() << "'\n";
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
    });
  }

  // Executor families, each carrying its own microarchitectural config.
  llvm::SmallDenseSet<llvm::StringRef> seenKinds;
  for (mlir::Attribute a : attr.getExecutors()) {
    auto fam = cast<perf::ExecutorFamilyAttr>(a);
    if (!seenKinds.insert(fam.getKind()).second) {
      llvm::errs() << "error: duplicate executor family kind '" << fam.getKind()
                   << "' in target '" << attr.getTargetName() << "'\n";
      return failure();
    }

    ExecutorFamilyModel fm;
    fm.kind = fam.getKind().str();
    fm.count = fam.getCount();
    fm.inOrder = fam.getInOrder();

    // Connectivity: every referenced space must be declared.
    for (mlir::Attribute s : fam.getSpaces()) {
      auto spaceName = cast<mlir::StringAttr>(s).getValue();
      if (!seenSpaces.contains(spaceName)) {
        llvm::errs() << "error: executor family '" << fam.getKind()
                     << "' references undeclared memory space '" << spaceName
                     << "' in target '" << attr.getTargetName() << "'\n";
        return failure();
      }
      fm.spaces.push_back(spaceName.str());
    }

    // Kind-to-config-type correspondence.
    mlir::Attribute cfg = fam.getConfig();
    if (auto dmaAttr = dyn_cast<perf::DmaAttr>(cfg)) {
      fm.dma = DmaConfig{
          .busBytesPerCycle = dmaAttr.getBusBytesPerCycle(),
          .fixedLatencyCycles = dmaAttr.getFixedLatencyCycles(),
          .maxOutstanding = dmaAttr.getMaxOutstanding(),
          .burstBytes = dmaAttr.getBurstBytes(),
      };
    } else if (auto arrayAttr = dyn_cast<perf::ArrayAttr>(cfg)) {
      fm.arrayConfig = ArrayConfig{
          .topology = arrayAttr.getTopology().str(),
          .meshRows = arrayAttr.getMeshRows(),
          .meshCols = arrayAttr.getMeshCols(),
          .tileRows = arrayAttr.getTileRows(),
          .tileCols = arrayAttr.getTileCols(),
          .peLatencyCycles = arrayAttr.getPeLatencyCycles(),
          .inputSkew = arrayAttr.getInputSkew(),
          .interTilePipelined = arrayAttr.getInterTilePipelined(),
      };
    } else {
      llvm::errs() << "error: executor family '" << fam.getKind()
                   << "' has unsupported config attribute (expected #perf.dma "
                      "or #perf.array) in target '"
                   << attr.getTargetName() << "'\n";
      return failure();
    }

    int64_t familyIndex = static_cast<int64_t>(model.families_.size());
    model.families_.push_back(std::move(fm));

    for (int64_t i = 0; i < fam.getCount(); ++i) {
      model.executors_.push_back({
          .kind = fam.getKind().str(),
          .index = i,
          .inOrder = fam.getInOrder(),
          .familyIndex = familyIndex,
      });
    }
  }

  for (mlir::Attribute a : attr.getLinks()) {
    auto link = cast<perf::LinkAttr>(a);
    for (int64_t i = 0; i < link.getCount(); ++i) {
      model.links_.push_back({
          .kind = link.getKind().str(),
          .index = i,
      });
    }
  }

  return model;
}

const MemSpaceConfig *TargetModel::findMemSpace(StringRef name) const {
  for (const auto &ms : memSpaces_) {
    if (ms.name == name)
      return &ms;
  }
  return nullptr;
}

const ExecutorFamilyModel *TargetModel::findFamily(StringRef kind) const {
  for (const auto &fam : families_) {
    if (fam.kind == kind)
      return &fam;
  }
  return nullptr;
}

const DmaConfig *TargetModel::dmaConfig(StringRef kind) const {
  const auto *fam = findFamily(kind);
  if (!fam || !fam->dma)
    return nullptr;
  return &*fam->dma;
}

const ArrayConfig *TargetModel::arrayConfig(StringRef kind) const {
  const auto *fam = findFamily(kind);
  if (!fam || !fam->arrayConfig)
    return nullptr;
  return &*fam->arrayConfig;
}

bool TargetModel::executorCanReach(StringRef kind, StringRef space) const {
  const auto *fam = findFamily(kind);
  return fam && fam->canReach(space);
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
