add_custom_target(PerfUnitTests)
set_target_properties(PerfUnitTests PROPERTIES FOLDER "Perf tests")

function(add_perf_unittest test_name)
  add_unittest(PerfUnitTests ${test_name} ${ARGN})
endfunction()

add_subdirectory(Analysis)
add_subdirectory(Sim)
