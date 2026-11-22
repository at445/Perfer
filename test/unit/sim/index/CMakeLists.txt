add_perf_unittest(ExecutionViewTest
  ExecutionViewTest.cpp
)

target_link_libraries(ExecutionViewTest PRIVATE
  MLIRPerfSim
  MLIRPerfDialect
  MLIRParser
  MLIRIR
  MLIRMemRefDialect
  MLIRFuncDialect
)
