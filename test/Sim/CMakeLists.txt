add_perf_unittest(TargetModelTest
  TargetModelTest.cpp
)

target_link_libraries(TargetModelTest PRIVATE
  MLIRPerfSim
  MLIRPerfDialect
  MLIRParser
  MLIRIR
)
