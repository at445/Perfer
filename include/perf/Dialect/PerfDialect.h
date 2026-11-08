// PerfDialect.h - perf dialect declaration
//
// The perf dialect describes compiler-produced execution plans for
// event-driven performance simulation. See perf-core-spec-v0.3.md and
// perf-simulator-architecture.md for the design contracts.

#ifndef PERF_DIALECT_PERFDIALECT_H
#define PERF_DIALECT_PERFDIALECT_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"

#include "perf/Dialect/PerfOpsDialect.h.inc"

#endif // PERF_DIALECT_PERFDIALECT_H
