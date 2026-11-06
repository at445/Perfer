// perf - parse, verify, and print MLIR (mlir-opt style).

#include "perf/Dialect/PerfAttrs.h"
#include "perf/Dialect/PerfDialect.h"

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
    llvm::cl::value_desc("program.mlir"));

static llvm::cl::opt<std::string> targetFilename(
    "target", llvm::cl::desc("Target config MLIR file"),
    llvm::cl::value_desc("target.mlir"));

static llvm::cl::opt<std::string> outputFilename(
    "o", llvm::cl::desc("Output filename ('-' for stdout)"),
    llvm::cl::value_desc("filename"), llvm::cl::init("-"));

//===----------------------------------------------------------------------===//
// Shared pipeline: parse + verify
//===----------------------------------------------------------------------===//

static int runPipeline(StringRef filename, MLIRContext &context) {
  std::string errorMessage;
  auto input = openInputFile(filename, &errorMessage);
  if (!input) {
    llvm::errs() << errorMessage << "\n";
    return 1;
  }

  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(input), llvm::SMLoc());
  SourceMgrDiagnosticHandler sourceMgrHandler(sourceMgr, &context);

  auto mod = parseSourceFile<ModuleOp>(sourceMgr, &context);
  if (!mod) {
    llvm::errs() << "error: failed to parse " << filename << "\n";
    return 1;
  }

  if (failed(verify(mod.get()))) {
    llvm::errs() << "error: verification failed for " << filename << "\n";
    return 1;
  }

  llvm::outs() << "OK: " << filename << "\n";
  return 0;
}

int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);
  llvm::cl::ParseCommandLineOptions(argc, argv, "Perfer perf simulator\n");

  DialectRegistry registry;
  registry.insert<perf::PerfDialect>();
  MLIRContext context(registry);
  context.loadAllAvailableDialects();

  if (!inputFilename.empty() && !targetFilename.empty()) {
    llvm::errs() << "error: --input and --target are mutually exclusive\n";
    return 1;
  }

  if (!inputFilename.empty())
    return runPipeline(inputFilename, context);
  if (!targetFilename.empty())
    return runPipeline(targetFilename, context);

  llvm::errs() << "error: either --input or --target is required\n";
  return 1;
}
