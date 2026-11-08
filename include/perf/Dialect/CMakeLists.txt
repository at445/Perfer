set(LLVM_TARGET_DEFINITIONS PerfDialect.td)
mlir_tablegen(PerfOpsDialect.h.inc -gen-dialect-decls -dialect=perf)
mlir_tablegen(PerfOpsDialect.cpp.inc -gen-dialect-defs -dialect=perf)

set(LLVM_TARGET_DEFINITIONS PerfAttrs.td)
mlir_tablegen(PerfAttrs.h.inc -gen-attrdef-decls -attrdefs-dialect=perf)
mlir_tablegen(PerfAttrs.cpp.inc -gen-attrdef-defs -attrdefs-dialect=perf)

set(LLVM_TARGET_DEFINITIONS PerfOps.td)
mlir_tablegen(PerfOps.h.inc -gen-op-decls)
mlir_tablegen(PerfOps.cpp.inc -gen-op-defs)
mlir_tablegen(PerfOpsTypes.h.inc -gen-typedef-decls -typedefs-dialect=perf)
mlir_tablegen(PerfOpsTypes.cpp.inc -gen-typedef-defs -typedefs-dialect=perf)

add_public_tablegen_target(MLIRPerfOpsIncGen)
