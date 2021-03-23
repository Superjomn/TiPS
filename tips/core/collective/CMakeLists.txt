core_gather_srcs(SRCS
        utils.cc
        coordinator.cc
        ops.cc
        )

cc_binary(test_collective_utils SRCS utils_test.cc DEPS tipscore)
target_compile_definitions(test_collective_utils PRIVATE _GLIBCXX_USE_CXX11_ABI=0)
mpi_test(test_collective_utils 5)
