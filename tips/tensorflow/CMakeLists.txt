core_gather_srcs(SRCS
        ops.cc)

add_test(NAME test_ops COMMAND mpirun -np 3 python3 ${CMAKE_CURRENT_SOURCE_DIR}/test_ops.py WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
