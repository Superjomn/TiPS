core_gather_srcs(SRCS
        table.cc
        client.cc
        server.cc
        dense_table.cc
        sparse_table.cc
        access_method.cc
        server_worker_route.cc
        dense_access_method.cc
        param.cc
        )

cc_test(test_dense_table SRCS dense_table_test.cc DEPS tipscore)
