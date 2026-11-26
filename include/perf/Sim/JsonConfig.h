// JsonConfig.h - Translate architect-level config.json into a TargetAttr.
//
// config.json is the architect-level graph of memory hierarchy / compute
// resources / interconnect (the form a GUI topology editor owns). It is one
// of two equivalent config encodings; the other is the simulator-level MLIR
// perf.target_def (#perf.target). Both lower to the same TargetAttr, after
// which the pipeline is identical.

#ifndef PERF_SIM_JSONCONFIG_H
#define PERF_SIM_JSONCONFIG_H

#include "perf/Dialect/PerfAttrs.h"

#include "mlir/IR/MLIRContext.h"

#include "llvm/Support/LogicalResult.h"

namespace perf {

// Parse a config.json file and build the equivalent #perf.target attribute
// in `context`. Returns failure (and prints to llvm::errs) on any malformed
// or missing field.
llvm::FailureOr<perf::TargetAttr>
targetAttrFromJsonConfig(llvm::StringRef filename,
                         mlir::MLIRContext &context);

} // namespace perf

#endif // PERF_SIM_JSONCONFIG_H
