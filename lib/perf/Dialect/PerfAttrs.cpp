// PerfAttrs.cpp - perf dialect attribute implementations

#include "perf/Dialect/PerfAttrs.h"
#include "perf/Dialect/PerfDialect.h"

#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace perf;

#define GET_ATTRDEF_CLASSES
#include "perf/Dialect/PerfAttrs.cpp.inc"

void PerfDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "perf/Dialect/PerfAttrs.cpp.inc"
      >();
}


