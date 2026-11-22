// perf - parse, verify, and simulate perf IR against a target config.

#include "perf/Analysis/ReachabilityVerifier.h"
#include "perf/Analysis/ScheduleLegalityVerifier.h"
#include "perf/Analysis/WriteConflictVerifier.h"
#include "perf/Dialect/PerfAttrs.h"
#include "perf/Dialect/PerfDialect.h"
#include "perf/Dialect/PerfOps.h"
#include "perf/Sim/TargetModel.h"

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
    llvm::cl::value_desc("program.mlir"), llvm::cl::Required);

static llvm::cl::opt<std::string> targetFilename(
    "target", llvm::cl::desc("Target design MLIR file (perf.target_def)"),
    llvm::cl::value_desc("design.mlir"), llvm::cl::Required);

static llvm::cl::opt<std::string> priorFilename(
    "prior",
    llvm::cl::desc("Simulation prior MLIR file (perf.prior_def). "
                   "Omit for an uncalibrated (identity) prior."),
    llvm::cl::value_desc("prior.mlir"), llvm::cl::init(""));

static llvm::cl::opt<std::string> outputFilename(
    "o", llvm::cl::desc("Output filename ('-' for stdout)"),
    llvm::cl::value_desc("filename"), llvm::cl::init("-"));

static llvm::cl::opt<bool> dumpContract(
    "dump-contract",
    llvm::cl::desc("Print the materialized compiler-simulator contract and exit"),
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

static llvm::FailureOr<perf::TargetModel> loadConfig(
    llvm::StringRef designFile, llvm::StringRef priorFile,
    MLIRContext &context) {
  auto designMod = parseModuleFile(designFile, context);
  if (!designMod)
    return failure();
  auto targetDef = findSingleTopLevel<perf::TargetDefOp>(*designMod,
                                                         designFile);
  if (failed(targetDef))
    return failure();

  perf::DesignAttr design = targetDef->getConfig();

  perf::PriorAttr prior;
  if (priorFile.empty()) {
    // Identity prior: zero calibration corrections. "No calibration data"
    // and "zero corrections" are the same thing physically; modelling it
    // as a real attr keeps the simulator free of optional branches.
    auto zeroCalibration = perf::CalibrationAttr::get(
        &context, /*extra_fill_cycles=*/0, /*extra_drain_cycles=*/0,
        /*extra_preload_cycles=*/0, /*extra_writeback_cycles=*/0,
        /*host_issue_cycles=*/0);
    prior = perf::PriorAttr::get(&context, design.getVersion(),
                                 "identity", zeroCalibration);
  } else {
    auto priorMod = parseModuleFile(priorFile, context);
    if (!priorMod)
      return failure();
    auto priorDef = findSingleTopLevel<perf::PriorDefOp>(*priorMod,
                                                         priorFile);
    if (failed(priorDef))
      return failure();
    prior = priorDef->getConfig();
  }

  auto modelOr = perf::TargetModel::build(design, prior);
  if (failed(modelOr)) {
    llvm::errs() << "error: failed to build TargetModel from " << designFile
                 << (priorFile.empty() ? " (identity prior)"
                                       : " + " + priorFile)
                 << "\n";
    return failure();
  }
  return modelOr;
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

  // Load design + prior configs
  auto targetOr = loadConfig(targetFilename, priorFilename, context);
  if (failed(targetOr))
    return 1;
  auto target = std::move(*targetOr);

  llvm::outs() << "Loaded design: " << target.getDesignAttr().getTargetName();
  if (priorFilename.empty())
    llvm::outs() << " (uncalibrated)\n";
  else
    llvm::outs() << " with prior: "
                 << target.getPriorAttr().getTargetName() << "\n";

  if (dumpContract) {
    auto contractMod = target.materializeContract(context);
    contractMod->print(llvm::outs());
    llvm::outs() << "\n";
    return 0;
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

  llvm::outs() << "OK: " << inputFilename << " against " << targetFilename << "\n";
  return 0;
}
