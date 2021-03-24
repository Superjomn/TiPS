#pragma once

#include <queue>
#include <sstream>
#include <unordered_map>
#include <vector>

#include "tips/core/collective/utils.h"
#include "tips/core/common/channel.h"
#include "tips/core/common/flatbuffers_utils.h"
#include "tips/core/common/managed_thread.h"
#include "tips/core/common/naive_rpc.h"
#include "tips/core/message/collective_messages_generated.h"

/**
 * The basic idea of the coordinator-based allreduce way is borrowed from Baidu-ring-reduce and Horovod.
 */

namespace tips {
namespace collective {

using CommunicationDoneCallback = std::function<void(StatusOr<tensorflow::Tensor>)>;

using RequestMessage  = FBS_TypeBufferOwned<message::RequestMessage>;
using ResponseMessage = FBS_TypeBufferOwned<message::ResponseMessage>;

RequestMessage CreateRequestMessage(int request_rank,
                                    message::RequestType request_type,
                                    message::DataType data_type,
                                    const std::string& tensor_name,
                                    const std::vector<int64_t>& tensor_shape);

/**
 * A wrapper to hold the RequestMessage flatbuffers object. It doesn't hold the data, leave the underlying buffer
 * managed externally.
 */
struct RequestMessageCloned {
  explicit RequestMessageCloned(const uint8_t* buffer) : buffer_(buffer) {}

  const collective::RequestMessage& msg() const {
    CHECK(buffer_);
    return *flatbuffers::GetRoot<collective::RequestMessage>(buffer_);
  }

 private:
  const uint8_t* buffer_;
};

struct OpRecord {
  // The rank performing this piece of op.
  int rank{-1};

  // The name of the op/tensor to be reduced.
  std::string name;

  OpKernelContext* op_context{};

  message::DataType dtype;

  // The input tensor.
  const Tensor* in_tensor;

  Tensor temp_tensor;

  // Allgather: vector of per-rank first-dimension sizes.
  std::vector<int64_t> sizes_vec;

  // The output tensor.
  Tensor* out_tensor;

  bool on_gpu{};

  // The callback to call after the op has completed.
  CommunicationDoneCallback callback;
};

/**
 * Table for storing Tensor metadata on rank zero. This is used for error checking and size calculation, as well as
 * determining when a reduction is ready to be done (when all nodes are ready to do it).
 */
using MessageTable = std::unordered_map<std::string, std::vector<RequestMessage>>;

/**
 * Table storing Tensors to be reduced, keyed by unique name.
 * This table contains everything necessary to do the reduction.
 */
using TensorTable = std::unordered_map<std::string, OpRecord>;

struct CollectiveState {
  //! A lock to guard all the shared resource.
  std::mutex mu;

  TensorTable tensor_table;

  std::shared_ptr<Channel<std::pair<RpcMsgHead, RequestMessage>>> message_queue{
      MakeChannel<std::pair<RpcMsgHead, RequestMessage>>()};

  ManagedThread background_thread;

  bool shut_down = false;

  //! Only exists on the coordinator node(rank = 0).
  std::unique_ptr<MessageTable> message_table;

  //! Singleton.
  static CollectiveState& Global() {
    static CollectiveState x;
    return x;
  }

  bool initialized() const;

  void Initialize();

  //! Tell the background thread to quit.
  void Finalize();

  ~CollectiveState() { background_thread.Terminate(); }
};

/**
 * Store the RequestMessage for a name and return whether the total count of RequestMessages for that tensor is now
 * equal to the MPI size (and thus we are ready to reduce the tensor).
 */
bool IncreTensorCount(MessageTable& table, RequestMessage&& msg, int mpi_size);

ResponseMessage ConstructResponseMessage(MessageTable& table, const std::string& name);

/**
 * Process an ResponseMessage by doing a reduction, a gather or raising an error.
 */
void PerformCollectiveOp(TensorTable& tensor_table,
                         message::ResponseType response_type,
                         const std::string name,
                         const std::string& error_msg);

// This function adds the op's record into the local op queue and sends a message to the coordinator indicating that
// this rank is ready to begin. The background thread will handle this message.
void EnqueueTensorCollective(const OpRecord& record, message::RequestType request_type);

inline bool IsCoordinator() { return mpi_rank() == 0; }

/**
 * The background thread logic.
 * NOTE this function should be triggered after MPI and RPC is initialized.
 */
void BackgroundThreadLoop();

void ShutdownBackgroundService();

}  // namespace collective
}  // namespace tips
