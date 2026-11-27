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

# Shared file-driven test helpers (test/unit/TestUtils.hpp).
target_include_directories(ExecutionViewTest PRIVATE
  ${CMAKE_SOURCE_DIR}/test/unit
)

# Fixture root: data/<feature>/input/program.mlir.
target_compile_definitions(ExecutionViewTest PRIVATE
  PERF_UNIT_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
)
