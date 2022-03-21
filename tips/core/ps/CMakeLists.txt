core_gather_srcs(SRCS
        table.cc
        dense_table.cc
        sparse_table.cc
        access_method.cc
        sparse_table_registry.cc
        #route.cc
        #dense_access_method.cc
        #param.cc
        #sparse_access_method.cc
        #ps_utils.cc
        #ps_server.cc
        #ps_client.cc
        )

#cc_test(test_dense_table SRCS dense_table_test.cc DEPS tipscore)
cc_test(test_access_method SRCS access_method_test.cc DEPS tipscore)
cc_test(test_sparse_table SRCS sparse_table_test.cc DEPS tipscore)

cc_binary(test_ps_server SRCS ps_server_test.cc DEPS tipscore)
mpi_test(test_ps_server 1)

cc_binary(test_ps_client SRCS ps_client_test.cc DEPS tipscore)
mpi_test(test_ps_client 6)

cc_test(test_sparse_table_registry SRCS sparse_table_registry_test.cc DEPS tipscore)
