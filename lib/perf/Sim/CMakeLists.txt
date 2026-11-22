add_mlir_library(MLIRPerfSim
  DependencyGraph.cpp
  ExecutionView.cpp
  ScheduleView.cpp
  TargetModel.cpp

  LINK_LIBS PUBLIC
  MLIRPerfDialect
  MLIRIR
)