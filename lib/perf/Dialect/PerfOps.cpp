// PerfOps.cpp - perf dialect op implementations (verifiers + MemoryEffects)

#include "perf/Dialect/PerfOps.h"
#include "perf/Dialect/PerfAttrs.h"
#include "perf/Dialect/PerfDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace perf;

#define GET_TYPEDEF_CLASSES
#include "perf/Dialect/PerfOpsTypes.cpp.inc"

#define GET_OP_CLASSES
#include "perf/Dialect/PerfOps.cpp.inc"

//===----------------------------------------------------------------------===//
// Helper: check memref space
//===----------------------------------------------------------------------===//

static LogicalResult verifyPerfSpaceMemref(Operation *op, Type type,
                                           StringRef name) {
  auto memrefTy = dyn_cast<MemRefType>(type);
  if (!memrefTy)
    return op->emitOpError("expected a memref type for ") << name;
  if (!memrefTy.hasStaticShape())
    return op->emitOpError("expected static shape for ") << name;
  auto spaceAttr = dyn_cast_if_present<perf::SpaceAttr>(memrefTy.getMemorySpace());
  if (!spaceAttr)
    return op->emitOpError("expected #perf.space memory space for ") << name;
  return success();
}

//===----------------------------------------------------------------------===//
// perf.target_def verifier
//===----------------------------------------------------------------------===//

LogicalResult TargetDefOp::verify() {
  // Structural checks of the config itself (resource families, capacities,
  // ports...) are enforced when TargetModel::build() consumes the attribute;
  // here we only ensure it exists and is well-typed (enforced by ODS).
  return success();
}

//===----------------------------------------------------------------------===//
// perf.alloc verifier
//===----------------------------------------------------------------------===//

LogicalResult AllocOp::verify() {
  return verifyPerfSpaceMemref(getOperation(), getResult().getType(), "result");
}

//===----------------------------------------------------------------------===//
// perf.dma_load / perf.dma_store verifier
//===----------------------------------------------------------------------===//

static LogicalResult verifyDmaBytes(Operation *op, int64_t bytes) {
  if (bytes <= 0)
    return op->emitOpError("expected bytes > 0");
  return success();
}

LogicalResult DmaLoadOp::verify() {
  if (failed(verifyDmaBytes(getOperation(), getBytes())))
    return failure();
  // dst is the on-chip buffer
  return verifyPerfSpaceMemref(getOperation(), getDst().getType(), "dst");
}

LogicalResult DmaStoreOp::verify() {
  if (failed(verifyDmaBytes(getOperation(), getBytes())))
    return failure();
  // src is the on-chip buffer
  return verifyPerfSpaceMemref(getOperation(), getSrc().getType(), "src");
}

//===----------------------------------------------------------------------===//
// perf.preload verifier
//===----------------------------------------------------------------------===//

LogicalResult PreloadOp::verify() {
  if (getM() <= 0 || getN() <= 0)
    return emitOpError("expected m > 0 and n > 0");

  StringRef role = getRole();
  if (role != "output" && role != "weight" && role != "input")
    return emitOpError("expected role in {output, weight, input}");

  return verifyPerfSpaceMemref(getOperation(), getDst().getType(), "dst");
}

//===----------------------------------------------------------------------===//
// perf.compute verifier
//===----------------------------------------------------------------------===//

LogicalResult ComputeOp::verify() {
  if (getM() <= 0 || getN() <= 0 || getK() <= 0)
    return emitOpError("expected m > 0, n > 0, k > 0");

  if (failed(verifyPerfSpaceMemref(getOperation(), getA().getType(), "a")))
    return failure();
  if (failed(verifyPerfSpaceMemref(getOperation(), getB().getType(), "b")))
    return failure();
  if (failed(verifyPerfSpaceMemref(getOperation(), getC().getType(), "c")))
    return failure();

  return success();
}

//===----------------------------------------------------------------------===//
// MemoryEffectsOpInterface
//===----------------------------------------------------------------------===//

void DmaLoadOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getOperation()->getOpOperand(0));
  effects.emplace_back(MemoryEffects::Write::get(), &getOperation()->getOpOperand(1));
}

void DmaStoreOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getOperation()->getOpOperand(0));
  effects.emplace_back(MemoryEffects::Write::get(), &getOperation()->getOpOperand(1));
}

void PreloadOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), &getOperation()->getOpOperand(0));
}

void ComputeOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getOperation()->getOpOperand(0));
  effects.emplace_back(MemoryEffects::Read::get(), &getOperation()->getOpOperand(1));
  effects.emplace_back(MemoryEffects::Read::get(), &getOperation()->getOpOperand(2));
  effects.emplace_back(MemoryEffects::Write::get(), &getOperation()->getOpOperand(2));
}

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

void PerfDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "perf/Dialect/PerfOpsTypes.cpp.inc"
      >();
}


