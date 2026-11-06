// PerfDialect.cpp - perf dialect implementation

#include "perf/Dialect/PerfDialect.h"
#include "perf/Dialect/PerfAttrs.h"

#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace perf;

#include "perf/Dialect/PerfOpsDialect.cpp.inc"

// registerAttributes lives in PerfAttrs.cpp because storage definitions
// are only visible there.
void PerfDialect::initialize() {
  registerAttributes();
}
