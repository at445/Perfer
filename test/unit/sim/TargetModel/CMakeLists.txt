add_perf_unittest(TargetModelTest
  TargetModelTest.cpp
)

target_link_libraries(TargetModelTest PRIVATE
  MLIRPerfSim
  MLIRPerfDialect
  MLIRParser
  MLIRIR
)

# Shared file-driven test helpers (test/unit/TestUtils.hpp).
target_include_directories(TargetModelTest PRIVATE
  ${CMAKE_SOURCE_DIR}/test/unit
)

# Fixture root: data/<feature>/{target.mlir, config.json, expected.dump}.
target_compile_definitions(TargetModelTest PRIVATE
  PERF_UNIT_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
)
