core_gather_srcs(SRCS
        utils.cc
        coordinator.cc
        )

cc_binary(test_collective_utils_allreduce SRCS utils_test.cc DEPS tipscore CXX_FLAGS -DTEST_ALLREDUCE)
mpi_test(test_collective_utils_allreduce 5)

cc_binary(test_collective_utils_allgather SRCS utils_test.cc DEPS tipscore CXX_FLAGS -DTEST_ALLGATHER)
mpi_test(test_collective_utils_allgather 3)

cc_binary(test_collective_utils_allgatherv SRCS utils_test.cc DEPS tipscore CXX_FLAGS -DTEST_ALLGATHERV)
mpi_test(test_collective_utils_allgatherv 3)

cc_binary(test_coordinator_allreduce SRCS coordinator_test.cc DEPS tipscore CXX_FLAGS -DTEST_ALLREDUCE)
mpi_test(test_collective_allreduce 3)

cc_binary(test_coordinator_allgather SRCS coordinator_test.cc DEPS tipscore CXX_FLAGS -DTEST_ALLGATHER)
mpi_test(test_collective_allgather 3)
