add_mlir_library(MLIRPerfAnalysis
  BufferProvenance.cpp
  ReachabilityVerifier.cpp
  ScheduleLegalityVerifier.cpp
  WriteConflictVerifier.cpp

  LINK_LIBS PUBLIC
  MLIRPerfDialect
  MLIRPerfSim
  MLIRIR
  MLIRMemRefDialect
)
