set(LLVM_TARGET_DEFINITIONS PerfDialect.td)
mlir_tablegen(PerfOpsDialect.h.inc -gen-dialect-decls -dialect=perf)
mlir_tablegen(PerfOpsDialect.cpp.inc -gen-dialect-defs -dialect=perf)

set(LLVM_TARGET_DEFINITIONS PerfAttrs.td)
mlir_tablegen(PerfAttrs.h.inc -gen-attrdef-decls -attrdefs-dialect=perf)
mlir_tablegen(PerfAttrs.cpp.inc -gen-attrdef-defs -attrdefs-dialect=perf)

add_public_tablegen_target(MLIRPerfOpsIncGen)
