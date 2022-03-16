core_gather_srcs(SRCS
        table.cc
        dense_table.cc
        sparse_table.cc
        # access_method.cc
        server_worker_route.cc
        #dense_access_method.cc
        #param.cc
        #sparse_access_method.cc
        server_service.cc
        ps_utils.cc
        table_service.cc
        )

cc_test(test_dense_table SRCS dense_table_test.cc DEPS tipscore)
cc_test(test_access_method SRCS access_method_test.cc DEPS tipscore)
cc_test(test_sparse_table SRCS sparse_table_test.cc DEPS tipscore)
