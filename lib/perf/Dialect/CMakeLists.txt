add_mlir_dialect_library(MLIRPerfDialect
  PerfAttrs.cpp
  PerfDialect.cpp

  DEPENDS
  MLIRPerfOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSideEffectInterfaces
)
