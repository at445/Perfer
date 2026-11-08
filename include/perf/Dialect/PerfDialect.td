// PerfDialect.td

#ifndef PERF_DIALECT_TD
#define PERF_DIALECT_TD

include "mlir/IR/DialectBase.td"

def PerfDialect : Dialect {
  let name = "perf";
  let cppNamespace = "::perf";
  let summary = "Compiled execution plan dialect for event-driven perf simulation.";
  let description = [{
    The perf dialect describes compiler-produced execution plans: static
    buffers, explicit DMA, array preload/compute and token dependencies.
    It contains no timing numbers; timing is derived by the simulator's
    CostModel from the target configuration.
  }];

  let useDefaultAttributePrinterParser = 1;
  let useDefaultTypePrinterParser = 1;

  let extraClassDeclaration = [{
    void registerAttributes();
    void registerTypes();
  }];
}

#endif // PERF_DIALECT_TD
