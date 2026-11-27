add_perf_unittest(ScheduleViewTest
  ScheduleViewTest.cpp
)

target_link_libraries(ScheduleViewTest PRIVATE
  MLIRPerfSim
  MLIRPerfDialect
  MLIRParser
  MLIRIR
  MLIRMemRefDialect
  MLIRFuncDialect
)

# Shared file-driven test helpers (test/unit/TestUtils.hpp).
target_include_directories(ScheduleViewTest PRIVATE
  ${CMAKE_SOURCE_DIR}/test/unit
)

# Fixture root: data/<feature>/input/program.mlir.
target_compile_definitions(ScheduleViewTest PRIVATE
  PERF_UNIT_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
)
