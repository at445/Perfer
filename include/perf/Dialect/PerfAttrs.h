// PerfAttrs.h - perf dialect attribute declarations

#ifndef PERF_DIALECT_PERFATTRS_H
#define PERF_DIALECT_PERFATTRS_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"

#include "perf/Dialect/PerfDialect.h"

#define GET_ATTRDEF_CLASSES
#include "perf/Dialect/PerfAttrs.h.inc"

#endif // PERF_DIALECT_PERFATTRS_H
