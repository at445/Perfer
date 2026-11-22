add_perf_unittest(BufferProvenanceTest
  BufferProvenanceTest.cpp
)

target_link_libraries(BufferProvenanceTest PRIVATE
  MLIRPerfAnalysis
  MLIRPerfDialect
  MLIRParser
  MLIRIR
  MLIRMemRefDialect
  MLIRFuncDialect
)
