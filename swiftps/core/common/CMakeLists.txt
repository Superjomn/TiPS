core_gather_srcs(SRCS
        common.cc
        managed_thread.cc
        thread_pool.cc
        semaphore.cc
        rpc.cc
        naive_buffer.cc
        )

cc_test(test_naive_buffer SRCS naive_buffer_test.cc DEPS swiftscore)
