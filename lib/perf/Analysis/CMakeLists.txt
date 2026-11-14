add_mlir_library(MLIRPerfAnalysis
  BufferProvenance.cpp

  LINK_LIBS PUBLIC
  MLIRPerfDialect
  MLIRIR
  MLIRMemRefDialect
)
