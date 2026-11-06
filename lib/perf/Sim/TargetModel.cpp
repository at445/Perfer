// TargetModel.cpp

#include "perf/Sim/TargetModel.h"

#include "mlir/IR/Attributes.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/MathExtras.h"

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
// SramConfig
//===----------------------------------------------------------------------===//

int64_t SramConfig::capacityBytes() const { return capacityKib * 1024; }

//===----------------------------------------------------------------------===//
// ResourceInstance
//===----------------------------------------------------------------------===//

std::string ResourceInstance::name() const {
  return kind + "#" + std::to_string(index);
}

//===----------------------------------------------------------------------===//
// TargetModel
//===----------------------------------------------------------------------===//

FailureOr<TargetModel> TargetModel::build(perf::TargetAttr attr) {
  TargetModel model(attr);

  auto dmaAttr = attr.getDma();
  model.dma_ = {
      .busBytesPerCycle = dmaAttr.getBusBytesPerCycle(),
      .fixedLatencyCycles = dmaAttr.getFixedLatencyCycles(),
      .maxOutstanding = dmaAttr.getMaxOutstanding(),
      .burstBytes = dmaAttr.getBurstBytes(),
  };

  auto arrayAttr = attr.getArray();
  model.array_ = {
      .topology = arrayAttr.getTopology().str(),
      .meshRows = arrayAttr.getMeshRows(),
      .meshCols = arrayAttr.getMeshCols(),
      .tileRows = arrayAttr.getTileRows(),
      .tileCols = arrayAttr.getTileCols(),
      .peLatencyCycles = arrayAttr.getPeLatencyCycles(),
      .inputSkew = arrayAttr.getInputSkew(),
      .interTilePipelined = arrayAttr.getInterTilePipelined(),
  };

  auto spadAttr = attr.getScratchpad();
  model.scratchpad_ = {
      .capacityKib = spadAttr.getCapacityKib(),
      .banks = spadAttr.getBanks(),
      .readPorts = spadAttr.getReadPorts(),
      .writePorts = spadAttr.getWritePorts(),
      .portWidthBytes = spadAttr.getPortWidthBytes(),
  };

  auto accAttr = attr.getAccumulator();
  model.accumulator_ = {
      .capacityKib = accAttr.getCapacityKib(),
      .banks = accAttr.getBanks(),
      .readPorts = accAttr.getReadPorts(),
      .writePorts = accAttr.getWritePorts(),
      .portWidthBytes = accAttr.getPortWidthBytes(),
  };

  for (mlir::Attribute a : attr.getResources()) {
    auto fam = cast<perf::ResourceFamilyAttr>(a);
    for (int64_t i = 0; i < fam.getCount(); ++i) {
      model.instances_.push_back({
          .kind = fam.getKind().str(),
          .index = i,
          .inOrder = fam.getInOrder(),
          .placeable = fam.getPlaceable(),
      });
    }
  }

  return model;
}

const ResourceInstance *TargetModel::findInstance(StringRef kind,
                                                   int64_t index) const {
  for (const auto &inst : instances_) {
    if (inst.kind == kind && inst.index == index)
      return &inst;
  }
  return nullptr;
}

const ResourceInstance *TargetModel::findInstance(StringRef name) const {
  for (const auto &inst : instances_) {
    if (inst.name() == name)
      return &inst;
  }
  return nullptr;
}
