core_gather_srcs(SRCS
        common.cc
        managed_thread.cc
        thread_pool.cc
        semaphore.cc
        swifts_rpc.cc
        naive_buffer.cc
        zmq_message.cc
        )

cc_test(test_naive_buffer SRCS naive_buffer_test.cc DEPS swiftscore)
cc_binary(test_swift_rpc SRCS swifts_rpc_test.cc DEPS swiftscore)
