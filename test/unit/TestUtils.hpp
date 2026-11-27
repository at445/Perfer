// TestUtils.h - Shared helpers for file-driven perf unit tests.
//
// Most perf UTs follow one pattern: parse fixture files (a #perf.target
// and/or a program input.mlir), build some structure, then compare either
// against an expected fixture or against another construction path. This
// header centralizes the boilerplate — MLIR context ownership, fixture file
// parsing, TargetModel construction, and dump capture — so each test file
// only carries the logic it actually varies.
//
// Fixture layout convention (PERF_UNIT_TEST_DATA_DIR points at the test's
// data root):
//   data/<feature>/target.mlir    simulator-level #perf.target
//   data/<feature>/config.json    architect-level config (lowers to target)
//   data/<feature>/input/         program .mlir files the feature consumes

#ifndef PERF_TEST_UNIT_TESTUTILS_HPP
#define PERF_TEST_UNIT_TESTUTILS_HPP

#include "perf/Dialect/PerfDialect.h"
#include "perf/Sim/JsonConfig.h"
#include "perf/Sim/TargetModel.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include "gtest/gtest.h"

#include <string>

namespace perftest {

// Owning MLIR context with the perf dialect loaded. Tests that need extra
// dialects (func, memref, ...) load them in their own SetUp.
class PerfTestBase : public ::testing::Test {
protected:
  void SetUp() override { context.loadDialect<perf::PerfDialect>(); }

  mlir::MLIRContext context;

  // Parse an MLIR module from inline text.
  mlir::ModuleOp parseModule(llvm::StringRef text) {
    return mlir::parseSourceString<mlir::ModuleOp>(text, &context).get();
  }

  // Parse an MLIR module from a fixture file; nullptr when unreadable.
  mlir::ModuleOp parseModuleFile(llvm::StringRef path) {
    auto bufferOr = llvm::MemoryBuffer::getFile(path);
    if (!bufferOr)
      return nullptr;
    return mlir::parseSourceString<mlir::ModuleOp>((*bufferOr)->getBuffer(),
                                                   &context)
        .get();
  }
};

// Read an entire fixture file into a string ("" on failure).
inline std::string readFile(llvm::StringRef path) {
  auto bufferOr = llvm::MemoryBuffer::getFile(path);
  if (!bufferOr)
    return "";
  return (*bufferOr)->getBuffer().str();
}

// Capture a model's dump() into a std::string for comparison.
inline std::string dumpToString(const perf::TargetModel &model) {
  std::string text;
  llvm::raw_string_ostream os(text);
  model.dump(os);
  return text;
}

// TargetModel-specific fixture helpers. The shared prior lives at the data
// root; each feature directory carries its own target.mlir / config.json.
class TargetModelTestBase : public PerfTestBase {
protected:
  // PERF_UNIT_TEST_DATA_DIR names the data root via CMake.
  const std::string dataDir = PERF_UNIT_TEST_DATA_DIR;

  std::string priorPath() const { return dataDir + "/prior.mlir"; }
  std::string featureDir(llvm::StringRef feature) const {
    return dataDir + "/" + feature.str();
  }
  std::string targetPath(llvm::StringRef feature) const {
    return featureDir(feature) + "/target.mlir";
  }
  std::string configPath(llvm::StringRef feature) const {
    return featureDir(feature) + "/config.json";
  }

  perf::PriorAttr loadPrior() {
    auto mod = parseModuleFile(priorPath());
    if (!mod)
      return nullptr;
    return llvm::dyn_cast_if_present<perf::PriorAttr>(
        mod->getAttr("perf.prior"));
  }

  // Build a TargetModel from a #perf.target attribute + the shared prior.
  llvm::FailureOr<perf::TargetModel>
  buildFromTargetAttr(perf::TargetAttr target) {
    auto prior = loadPrior();
    if (!prior)
      return llvm::failure();
    return perf::TargetModel::build(target, prior);
  }

  // Build from a feature directory's target.mlir.
  llvm::FailureOr<perf::TargetModel>
  buildFromFeatureTarget(llvm::StringRef feature) {
    auto mod = parseModuleFile(targetPath(feature));
    if (!mod)
      return llvm::failure();
    auto target = llvm::dyn_cast_if_present<perf::TargetAttr>(
        mod->getAttr("perf.target"));
    if (!target)
      return llvm::failure();
    return buildFromTargetAttr(target);
  }

  // Build from a feature directory's config.json (via the json projection).
  llvm::FailureOr<perf::TargetModel>
  buildFromFeatureConfig(llvm::StringRef feature) {
    auto attr = perf::targetAttrFromJsonConfig(configPath(feature),
                                                     context);
    if (mlir::failed(attr))
      return llvm::failure();
    return buildFromTargetAttr(*attr);
  }

  // Build from inline target text + the shared file-based prior.
  llvm::FailureOr<perf::TargetModel>
  buildFromText(llvm::StringRef targetText) {
    auto mod = parseModule(targetText);
    if (!mod)
      return llvm::failure();
    auto target = llvm::dyn_cast_if_present<perf::TargetAttr>(
        mod->getAttr("perf.target"));
    if (!target)
      return llvm::failure();
    return buildFromTargetAttr(target);
  }
};

} // namespace perftest

#endif // PERF_TEST_UNIT_TESTUTILS_HPP
