add_mlir_library(MLIRPerfSim
  TargetModel.cpp

  LINK_LIBS PUBLIC
  MLIRPerfDialect
  MLIRIR
)