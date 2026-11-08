get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

add_llvm_executable(perf
  perf.cpp
)

llvm_update_compile_flags(perf)

target_link_libraries(perf PRIVATE
  MLIRPerfDialect
  MLIRPerfSim
  MLIRParser
  MLIRFuncDialect
  MLIRSupport
  MLIRIR
)

mlir_check_all_link_libraries(perf)
