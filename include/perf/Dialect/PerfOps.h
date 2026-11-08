// PerfOps.h - perf dialect op declarations

#ifndef PERF_DIALECT_PERFOPS_H
#define PERF_DIALECT_PERFOPS_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "perf/Dialect/PerfDialect.h"
#include "perf/Dialect/PerfAttrs.h"

#define GET_TYPEDEF_CLASSES
#include "perf/Dialect/PerfOpsTypes.h.inc"

#define GET_OP_CLASSES
#include "perf/Dialect/PerfOps.h.inc"

#endif // PERF_DIALECT_PERFOPS_H
