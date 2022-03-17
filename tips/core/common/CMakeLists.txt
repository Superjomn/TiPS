core_gather_srcs(SRCS
        common.cc
        managed_thread.cc
        thread_pool.cc
        semaphore.cc
        naive_rpc.cc
        naive_buffer.cc
        zmq_message.cc
        semaphore.cc
        channel.cc
        flatbuffers_utils.cc
        logging.cc
        thread_group.cc
        rwlock.cc
        vec.cc
        any_vec.cc
        datatype.cc
        )

cc_test(test_naive_buffer SRCS naive_buffer_test.cc DEPS tipscore)
cc_test(test_channel SRCS channel_test.cc DEPS tipscore)
target_link_libraries(test_channel ${Tensorflow_LIBRARIES})

cc_binary(test_naive_rpc SRCS naive_rpc_test.cc DEPS tipscore)
mpi_test(test_naive_rpc 3)

cc_test(test_flatbuffers_utils SRCS flatbuffers_utils_test.cc DEPS tipscore)
cc_test(test_thread_group SRCS thread_group_test.cc DEPS tipscore)
cc_test(test_zmq_message SRCS zmq_message_test.cc DEPS tipscore)
cc_test(test_any_vec SRCS any_vec_test.cc DEPS tipscore)
