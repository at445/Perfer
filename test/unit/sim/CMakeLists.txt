# sim UTs are split by concern:
#   TargetModel/  TargetModel construction + dump (fixture-driven)
#   OpView/       op-level views (ExecutionView / DependencyGraph / ScheduleView)
add_subdirectory(TargetModel)
add_subdirectory(OpView)
