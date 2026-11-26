// perf - parse, verify, and simulate perf IR against a target config.

#include "perf/Analysis/ArrayStateVerifier.h"
#include "perf/Analysis/ReachabilityVerifier.h"
#include "perf/Analysis/ScheduleLegalityVerifier.h"
#include "perf/Analysis/WriteConflictVerifier.h"
#include "perf/Dialect/PerfAttrs.h"
#include "perf/Dialect/PerfDialect.h"
#include "perf/Dialect/PerfOps.h"
#include "perf/Sim/TargetModel.h"
#include "perf/Sim/JsonConfig.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Command-line options
//===----------------------------------------------------------------------===//

static llvm::cl::opt<std::string> inputFilename(
    "input", llvm::cl::desc("Input MLIR program file"),
    llvm::cl::value_desc("program.mlir"), llvm::cl::init(""));

static llvm::cl::opt<std::string> targetFilename(
    "target",
    llvm::cl::desc("Simulator-level target MLIR file (perf.target_def), the "
                   "executor/link vocabulary the simulator consumes directly"),
    llvm::cl::value_desc("target.mlir"), llvm::cl::init(""));

static llvm::cl::opt<std::string> configFilename(
    "config",
    llvm::cl::desc("Architect-level config JSON file (memory hierarchy / "
                   "compute resources / interconnect graph). Projected to a "
                   "target internally (equivalent to --dump-target + simulate)"),
    llvm::cl::value_desc("config.json"), llvm::cl::init(""));

static llvm::cl::opt<std::string> priorFilename(
    "prior",
    llvm::cl::desc("Simulation prior MLIR file (perf.prior_def). "
                   "Omit for an uncalibrated (identity) prior."),
    llvm::cl::value_desc("prior.mlir"), llvm::cl::init(""));

static llvm::cl::opt<std::string> outputFilename(
    "o", llvm::cl::desc("Output filename ('-' for stdout)"),
    llvm::cl::value_desc("filename"), llvm::cl::init("-"));

static llvm::cl::opt<bool> dumpTarget(
    "dump-target",
    llvm::cl::desc("Project the config to the simulator-level target and "
                   "print it, then exit"),
    llvm::cl::init(false));

//===----------------------------------------------------------------------===//
// Parse helpers
//===----------------------------------------------------------------------===//

static OwningOpRef<ModuleOp> parseModuleFile(StringRef filename,
                                             MLIRContext &context) {
  std::string errorMessage;
  auto input = openInputFile(filename, &errorMessage);
  if (!input) {
    llvm::errs() << errorMessage << "\n";
    return nullptr;
  }

  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(input), llvm::SMLoc());
  SourceMgrDiagnosticHandler handler(sourceMgr, &context);

  auto mod = parseSourceFile<ModuleOp>(sourceMgr, &context);
  if (!mod) {
    llvm::errs() << "error: failed to parse " << filename << "\n";
    return nullptr;
  }

  if (failed(verify(mod.get()))) {
    llvm::errs() << "error: verification failed for " << filename << "\n";
    return nullptr;
  }

  return mod;
}

//===----------------------------------------------------------------------===//
// Config loading
//===----------------------------------------------------------------------===//

// Find the unique top-level op of type OpTy in a module.
template <typename OpTy>
static llvm::FailureOr<OpTy> findSingleTopLevel(ModuleOp mod,
                                                llvm::StringRef filename) {
  OpTy result;
  for (auto &op : mod.getBody()->getOperations()) {
    if (auto match = dyn_cast<OpTy>(&op)) {
      if (result) {
        llvm::errs() << "error: multiple " << OpTy::getOperationName()
                     << " ops in " << filename << "\n";
        return failure();
      }
      result = match;
    }
  }
  if (!result) {
    llvm::errs() << "error: no " << OpTy::getOperationName() << " found in "
                 << filename << "\n";
    return failure();
  }
  return result;
}

// Resolve the TargetAttr from exactly one of --config (architect-level JSON)
// or --target (simulator-level MLIR perf.target_def).
static llvm::FailureOr<perf::TargetAttr> resolveTargetAttr(
    llvm::StringRef configFile, llvm::StringRef targetFile,
    MLIRContext &context) {
  if (configFile.empty() == targetFile.empty()) {
    llvm::errs() << "error: exactly one of --config or --target is required\n";
    return failure();
  }

  if (!configFile.empty())
    return perf::targetAttrFromJsonConfig(configFile, context);

  auto targetMod = parseModuleFile(targetFile, context);
  if (!targetMod)
    return failure();
  auto targetDef = findSingleTopLevel<perf::TargetDefOp>(*targetMod,
                                                         targetFile);
  if (failed(targetDef))
    return failure();
  return targetDef->getConfig();
}

// Resolve the simulation prior, defaulting to the identity (uncalibrated)
// prior when --prior is omitted.
static llvm::FailureOr<perf::PriorAttr> resolvePrior(
    llvm::StringRef priorFile, perf::TargetAttr target,
    MLIRContext &context) {
  if (priorFile.empty()) {
    // Identity prior: zero calibration corrections. "No calibration data"
    // and "zero corrections" are the same thing physically; modelling it
    // as a real attr keeps the simulator free of optional branches.
    auto zeroCalibration = perf::CalibrationAttr::get(
        &context, /*extra_fill_cycles=*/0, /*extra_drain_cycles=*/0,
        /*extra_preload_cycles=*/0, /*extra_writeback_cycles=*/0,
        /*host_issue_cycles=*/0);
    return perf::PriorAttr::get(&context, target.getVersion(),
                                "identity", zeroCalibration);
  }
  auto priorMod = parseModuleFile(priorFile, context);
  if (!priorMod)
    return failure();
  auto priorDef = findSingleTopLevel<perf::PriorDefOp>(*priorMod,
                                                       priorFile);
  if (failed(priorDef))
    return failure();
  return priorDef->getConfig();
}

//===----------------------------------------------------------------------===//
// Main pipeline
//===----------------------------------------------------------------------===//

int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);
  llvm::cl::ParseCommandLineOptions(argc, argv, "Perfer perf simulator\n");

  DialectRegistry registry;
  registry.insert<perf::PerfDialect>();
  registry.insert<func::FuncDialect>();
  MLIRContext context(registry);
  context.loadAllAvailableDialects();

  // Resolve the simulator-level target from --config / --target.
  auto targetAttrOr = resolveTargetAttr(configFilename, targetFilename,
                                        context);
  if (failed(targetAttrOr))
    return 1;
  auto targetAttr = *targetAttrOr;

  // --dump-target: print the resolved target as a perf.target_def and exit.
  // This is a pure projection (config.json -> target.mlir); no TargetModel
  // is built. A malformed target surfaces later through the verifier rules.
  if (dumpTarget) {
    auto mod = mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));
    mlir::OpBuilder builder(&context);
    builder.setInsertionPointToStart(mod.getBody());
    perf::TargetDefOp::create(builder, builder.getUnknownLoc(),
                              targetAttr.getTargetName(), targetAttr);
    mod->print(llvm::outs());
    llvm::outs() << "\n";
    return 0;
  }

  // Simulation path: build the TargetModel (one-way, target -> model).
  auto priorOr = resolvePrior(priorFilename, targetAttr, context);
  if (failed(priorOr))
    return 1;
  auto prior = *priorOr;

  auto modelOr = perf::TargetModel::build(targetAttr, prior);
  if (failed(modelOr))
    return 1;
  auto target = std::move(*modelOr);

  llvm::outs() << "Loaded target: " << targetAttr.getTargetName();
  if (priorFilename.empty())
    llvm::outs() << " (uncalibrated)\n";
  else
    llvm::outs() << " with prior: "
                 << target.getPriorAttr().getTargetName() << "\n";

  // Simulation needs the program IR.
  if (inputFilename.empty()) {
    llvm::errs() << "error: --input is required unless --dump-target is given\n";
    return 1;
  }

  // Parse program
  auto mod = parseModuleFile(inputFilename, context);
  if (!mod)
    return 1;

  // R1 reachability: every op's memref operands must be reachable from its
  // placement resource, per the projected contract.
  if (failed(perf::verifyReachability(*mod, target)))
    return 1;

  // R3 schedule legality: per-resource total order must be a linear
  // extension of the token partial order.
  if (failed(perf::verifyScheduleLegality(*mod)))
    return 1;

  // R4 write conflicts: conflicting accesses with at least one write must
  // be ordered by the token partial order.
  if (failed(perf::verifyWriteConflict(*mod)))
    return 1;

  // R6 array PE-state provenance: compute's PE state must trace back to
  // array_init or an earlier compute along the token chain.
  if (failed(perf::verifyArrayState(*mod)))
    return 1;

  llvm::StringRef cfg = configFilename.empty()
                            ? llvm::StringRef(targetFilename)
                            : llvm::StringRef(configFilename);
  llvm::outs() << "OK: " << inputFilename << " against " << cfg << "\n";
  return 0;
}
