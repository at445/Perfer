// perf - parse, verify, and simulate perf IR against a target config.

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
    "target", llvm::cl::desc("Target config MLIR file"),
    llvm::cl::value_desc("target.mlir"), llvm::cl::Required);

static llvm::cl::opt<std::string> outputFilename(
    "o", llvm::cl::desc("Output filename ('-' for stdout)"),
    llvm::cl::value_desc("filename"), llvm::cl::init("-"));

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
// Target loading
//===----------------------------------------------------------------------===//

static llvm::FailureOr<perf::TargetModel> loadTarget(StringRef filename,
                                                      MLIRContext &context) {
  auto mod = parseModuleFile(filename, context);
  if (!mod)
    return failure();

  // Expect a single perf.target_def op at top level.
  perf::TargetDefOp targetDef;
  for (auto &op : mod->getBody()->getOperations()) {
    if (auto td = dyn_cast<perf::TargetDefOp>(&op)) {
      if (targetDef) {
        llvm::errs() << "error: multiple perf.target_def ops in " << filename << "\n";
        return failure();
      }
      targetDef = td;
    }
  }
  if (!targetDef) {
    llvm::errs() << "error: no perf.target_def found in " << filename << "\n";
    return failure();
  }

  auto modelOr = perf::TargetModel::build(targetDef.getConfig());
  if (failed(modelOr)) {
    llvm::errs() << "error: failed to build TargetModel from " << filename << "\n";
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

  // Load target config
  auto targetOr = loadTarget(targetFilename, context);
  if (failed(targetOr))
    return 1;
  auto target = std::move(*targetOr);

  llvm::outs() << "Loaded target: " << target.getAttr().getTargetName() << "\n";

  // Parse program
  auto mod = parseModuleFile(inputFilename, context);
  if (!mod)
    return 1;

  llvm::outs() << "OK: " << inputFilename << " against " << targetFilename << "\n";
  return 0;
}
