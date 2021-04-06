core_gather_srcs(SRCS
        table.cc
        client.cc
        ps_server.cc
        dense_table.cc
        sparse_table.cc
        access_method.cc
        route.cc
        dense_access_method.cc
        param.cc
        sparse_access_method.cc
        )

cc_test(test_dense_table SRCS dense_table_test.cc DEPS tipscore)
cc_test(test_access_method SRCS access_method_test.cc DEPS tipscore)
cc_test(test_sparse_table SRCS sparse_table_test.cc DEPS tipscore)
cc_binary(test_ps_server SRCS ps_server_test.cc DEPS tipscore)
