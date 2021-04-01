core_gather_srcs(SRCS
        tips_mpi.cc
        mpi_context.cc
        mpi_group.cc
        )

cc_binary(test_mpi_allreduce SRCS mpi_allreduce_test.cc DEPS tipscore)
add_test(NAME test_mpi_allreduce_main COMMAND mpirun -np 3 --allow-run-as-root test_mpi_allreduce)

cc_binary(test_mpi_allgather SRCS mpi_allgather_test.cc DEPS tipscore)
add_test(NAME test_mpi_allgather_main COMMAND mpirun -np 3 --allow-run-as-root test_mpi_allgather)

cc_binary(test_mpi_group SRCS mpi_group_test.cc DEPS tipscore)
add_test(NAME test_mpi_group_main COMMAND mpirun -np 4 --allow-run-as-root test_mpi_group)
