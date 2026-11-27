# op-view UTs are split by component, each with its own fixtures under
# data/<feature>/input/program.mlir:
#   executionView/    the executor-op index (OpId, placement, token producers)
#   dependencyGraph/  token dependency edges and readiness tracking
#   scheduleView/     per-resource execution order
add_subdirectory(executionView)
add_subdirectory(dependencyGraph)
add_subdirectory(scheduleView)
