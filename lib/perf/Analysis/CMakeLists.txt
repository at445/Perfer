add_mlir_library(MLIRPerfAnalysis
  BufferProvenance.cpp
  ReachabilityVerifier.cpp

  LINK_LIBS PUBLIC
  MLIRPerfDialect
  MLIRIR
  MLIRMemRefDialect
)
