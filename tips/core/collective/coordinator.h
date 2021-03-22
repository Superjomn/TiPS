#pragma once

#include <queue>
#include <sstream>
#include <unordered_map>
#include <vector>

#include "tips/core/collective/utils.h"
#include "tips/core/common/channel.h"
#include "tips/core/common/managed_thread.h"
#include "tips/core/common/naive_rpc.h"
#include "tips/core/message/collective_messages_generated.h"

/**
 * The basic idea of the coordinator-based allreduce way is borrowed from Baidu-ring-reduce and Horovod.
 */

namespace tips {
namespace collective {

using CommunicationDoneCallback = std::function<void(StatusOr<tensorflow::Tensor>)>;

/**
 * A wrapper to copy the RequestMessage flatbuffers object. It allocate and copy the data itself.
 */
template <typename FBS_T>
struct FBS_TypeBufferOwned {
  FBS_TypeBufferOwned() = default;
  FBS_TypeBufferOwned(const FBS_TypeBufferOwned& other) { Copy(other.buffer_, other.len_); }
  FBS_TypeBufferOwned(flatbuffers::DetachedBuffer&& buffer)
      : detached_buffer_(new flatbuffers::DetachedBuffer(std::move(buffer))) {}
  FBS_TypeBufferOwned& operator=(FBS_TypeBufferOwned&& other) {
    // Clear this.
    if (buffer_) {
      delete buffer_;
    }
    detached_buffer_.reset(nullptr);

    // Set the new data
    if (other.buffer_) {
      buffer_       = other.buffer_;
      len_          = other.len_;
      other.buffer_ = nullptr;
      other.len_    = 0;
    }

    if (other.detached_buffer_) {
      detached_buffer_ = std::move(other.detached_buffer_);
    }

    return *this;
  }

  /**
   * construct.
   * @param buffer address of the buffer.
   * @param len length of the buffer.
   * @param need_copy whether need to allocate memory and copy the data, false will just copy the pointer.
   */
  FBS_TypeBufferOwned(uint8_t* buffer, size_t len, bool need_copy = true) {
    if (need_copy) {
      Copy(buffer, len);
    } else {
      buffer_ = buffer;
      len_    = len;
    }
  }

  FBS_TypeBufferOwned(FBS_TypeBufferOwned&& other)
      : buffer_(other.buffer_), len_(other.len_), detached_buffer_(std::move(other.detached_buffer_)) {
    other.buffer_ = nullptr;
    other.len_    = 0;
  }

  const FBS_T& msg() const {
    CHECK(buffer_ || detached_buffer_);
    if (buffer_) {
      return *flatbuffers::GetRoot<FBS_T>(buffer_);
    } else {
      return *flatbuffers::GetRoot<FBS_T>(detached_buffer_->data());
    }
  }

  ~FBS_TypeBufferOwned() {
    if (buffer_) {
      free(buffer_);
    }
  }

 private:
  void Copy(const uint8_t* buffer, size_t len);

  uint8_t* buffer_{};
  size_t len_{};

  std::unique_ptr<flatbuffers::DetachedBuffer> detached_buffer_;
};

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

  // Allgather: vector of per-rank first-dimension sizes.
  std::vector<int64_t> sizes_vec;

  // The output tensor.
  Tensor* out_tensor;

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

  std::shared_ptr<Channel<RequestMessage>> message_queue{MakeChannel<RequestMessage>()};

  ManagedThread background_thread;

  bool shut_down = false;

  //! Only exists on the coordinator node(rank = 0).
  std::unique_ptr<MessageTable> message_table;

  //! Singleton.
  static CollectiveState& Global() {
    static CollectiveState x;
    return x;
  }

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
void PerformCollectiveOp(TensorTable& tensor_table, ResponseMessage&& response);

// This function adds the op's record into the local op queue and sends a message to the coordinator indicating that
// this rank is ready to begin. The background thread will handle this message.
void EnqueueTensorCollective(const OpRecord& record, message::RequestType request_type);

bool IsCoordinator() { return mpi_rank() == 0; }

void BackgroundThreadLoop() {
  auto& message_queue = CollectiveState::Global().message_queue;
  auto& message_table = CollectiveState::Global().message_table;
  if (IsCoordinator()) {
    CollectiveState::Global().message_table.reset(new MessageTable);

    bool should_shutdown = false;

    std::queue<std::string> ready_to_reduce;

    do {
      // process all the message in the queue
      RequestMessage message;
      if (message_queue->Read(&message)) {
        if (!IsCoordinator()) {  // send message to coordinator
          // TODO
        } else {
          auto tensor_name = message.msg().tensor_name()->str();
          bool reduce      = IncreTensorCount(*message_table, std::move(message), mpi_size());
          if (reduce) {
            ready_to_reduce.push(tensor_name);
          }
        }
      }

      if (IsCoordinator()) {
        for (int i = 0; i < ready_to_reduce.size(); i++) {
          auto name = ready_to_reduce.front();
          ready_to_reduce.pop();

          ResponseMessage response = ConstructResponseMessage(*message_table, name);
          // TODO send the response back
          PerformCollectiveOp(CollectiveState::Global().tensor_table, std::move(response));
        }
      }

    } while (!should_shutdown);
  }
}

template <typename FBS_T>
void FBS_TypeBufferOwned<FBS_T>::Copy(const uint8_t* buffer, size_t len) {
  CHECK(!buffer_) << "FBS_TypeBufferOwned duplicate assign found";
  if (len == 0) return;
  len_    = len;
  buffer_ = reinterpret_cast<uint8_t*>(std::malloc(len));
  CHECK(buffer_) << "allocate memory failed";
  std::memcpy(buffer_, buffer, len);
}

}  // namespace collective
}  // namespace tips
