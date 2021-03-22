core_gather_srcs(SRCS
        tips_mpi.cc
        mpi_context.cc
        )

cc_binary(test_mpi_allreduce SRCS mpi_allreduce_test.cc DEPS tipscore)

add_test(NAME test_mpi_allreduce_main COMMAND mpirun -np 3 test_mpi_allreduce)
