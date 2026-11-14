add_perf_unittest(ReachabilityVerifierTest
  ReachabilityVerifierTest.cpp
)

target_link_libraries(ReachabilityVerifierTest PRIVATE
  MLIRPerfAnalysis
  MLIRPerfDialect
  MLIRParser
  MLIRIR
  MLIRMemRefDialect
  MLIRFuncDialect
)
