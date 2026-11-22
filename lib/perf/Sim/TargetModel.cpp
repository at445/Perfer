// TargetModel.cpp

#include "perf/Sim/TargetModel.h"

#include "perf/Dialect/PerfOps.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
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

//===----------------------------------------------------------------------===//
// TargetModel
//===----------------------------------------------------------------------===//

namespace {

DmaConfig toDmaConfig(perf::DmaAttr a) {
  return {
      .busBytesPerCycle = a.getBusBytesPerCycle(),
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
  };
}

} // namespace

FailureOr<TargetModel> TargetModel::build(perf::DesignAttr design,
                                          perf::PriorAttr prior) {
  TargetModel model(design, prior);

  // Named memory spaces: the IDs that program-side #perf.space<name, ...>
  // attributes resolve against.
  llvm::SmallDenseSet<llvm::StringRef> seenSpaces;
  llvm::SmallVector<std::string> allSpaces;
  for (mlir::Attribute a : design.getMemorySpaces()) {
    auto ms = cast<perf::MemSpaceAttr>(a);
    if (!seenSpaces.insert(ms.getName()).second) {
      llvm::errs() << "error: duplicate memory space name '" << ms.getName()
                   << "' in design '" << design.getTargetName() << "'\n";
      return failure();
    }
    allSpaces.push_back(ms.getName().str());
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

  // Project the physical design parameters into the compiler-simulator
  // contract. Every executor reaches every declared memory space (R1
  // connectivity); finer-grained topology belongs to a later milestone.
  auto addFamily = [&](llvm::StringRef kind, int64_t count,
                       std::optional<DmaConfig> dma,
                       std::optional<ArrayConfig> array) {
    if (count <= 0)
      return;
    ExecutorFamilyModel fm;
    fm.kind = kind.str();
    fm.count = count;
    fm.inOrder = true;  // executors are in-order resources (default order)
    fm.spaces = allSpaces;
    fm.dma = std::move(dma);
    fm.arrayConfig = std::move(array);

    int64_t familyIndex = static_cast<int64_t>(model.families_.size());
    model.families_.push_back(std::move(fm));
    for (int64_t i = 0; i < count; ++i) {
      model.executors_.push_back({
          .kind = kind.str(),
          .index = i,
          .inOrder = true,
          .familyIndex = familyIndex,
      });
    }
  };

  auto dmaCfg = toDmaConfig(design.getDmaChannel());
  addFamily("dma_read", design.getDmaReadPorts(), dmaCfg, std::nullopt);
  addFamily("dma_write", design.getDmaWritePorts(), dmaCfg, std::nullopt);
  addFamily("array", design.getArrays(), std::nullopt,
            toArrayConfig(design.getArrayUarch()));

  if (design.getShareDramBus())
    model.links_.push_back({.kind = "dram_bus", .index = 0});

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

//===----------------------------------------------------------------------===//
// Contract materialization
//===----------------------------------------------------------------------===//

mlir::OwningOpRef<mlir::ModuleOp>
TargetModel::materializeContract(mlir::MLIRContext &context) const {
  auto loc = mlir::UnknownLoc::get(&context);
  auto module = mlir::ModuleOp::create(loc);
  mlir::OpBuilder builder(&context);
  builder.setInsertionPointToStart(module.getBody());

  // Group link instances back into (kind, count) families for emission.
  llvm::SmallVector<mlir::Attribute> linkAttrs;
  {
    llvm::StringMap<int64_t> linkCounts;
    llvm::SmallVector<std::string> linkOrder;
    for (const auto &l : links_) {
      if (linkCounts[l.kind]++ == 0)
        linkOrder.push_back(l.kind);
    }
    for (const auto &kind : linkOrder) {
      linkAttrs.push_back(perf::LinkAttr::get(
          &context, kind, linkCounts[kind]));
    }
  }

  // Executor families: one #perf.executor_family per family, reusing the
  // original microarchitecture attributes from the design.
  llvm::SmallVector<mlir::Attribute> execAttrs;
  for (const auto &fam : families_) {
    mlir::Attribute config;
    if (fam.dma)
      config = designAttr_.getDmaChannel();
    else
      config = designAttr_.getArrayUarch();

    llvm::SmallVector<mlir::Attribute> spaces;
    for (const auto &s : fam.spaces)
      spaces.push_back(builder.getStringAttr(s));

    execAttrs.push_back(perf::ExecutorFamilyAttr::get(
        &context, fam.kind, fam.count, fam.inOrder,
        builder.getArrayAttr(spaces), config));
  }

  auto contractAttr = perf::ContractAttr::get(
      &context,
      designAttr_.getVersion(),
      designAttr_.getTargetName(),
      designAttr_.getFrequencyMhz(),
      designAttr_.getMemorySpaces(),
      builder.getArrayAttr(execAttrs),
      builder.getArrayAttr(linkAttrs));

  builder.create<perf::ContractDefOp>(loc, designAttr_.getTargetName(),
                                      contractAttr);
  return module;
}
