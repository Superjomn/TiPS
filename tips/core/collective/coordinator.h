#pragma once

#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/op_kernel.h>
#include <tensorflow/core/framework/shape_inference.h>
#include <tensorflow/stream_executor/lib/statusor.h>
#include <sstream>
#include <unordered_map>
#include <vector>

#include "tips/core/common/naive_rpc.h"
#include "tips/core/message/collective_messages_generated.h"

/**
 * The basic idea of the coordinator-based allreduce way is borrowed from Baidu-ring-reduce and Horovod.
 */

namespace tips {
namespace collective {

template <typename T>
using StatusOr = stream_executor::port::StatusOr<T>;

/**
 * A wrapper to copy the RequestMessage flatbuffers object. It allocate and copy the data itself.
 */
template <typename FBS_T>
struct FBS_TypeBufferOwned {
  FBS_TypeBufferOwned() = delete;
  FBS_TypeBufferOwned(const FBS_TypeBufferOwned& other) { Copy(other.buffer_, other.len_); }

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

  FBS_TypeBufferOwned(FBS_TypeBufferOwned&& other) : buffer_(other.buffer_), len_(other.len_) {
    other.buffer_ = nullptr;
    other.len_    = 0;
  }

  const FBS_T& msg() const {
    CHECK(buffer_);
    return *flatbuffers::GetRoot<collective::RequestMessage>(buffer_);
  }

  ~FBS_TypeBufferOwned() {
    if (buffer_) free(buffer_);
  }

 private:
  void Copy(const uint8_t* buffer, size_t len);

  uint8_t* buffer_{};
  size_t len_{};
};

using RequestMessage  = FBS_TypeBufferOwned<message::RequestMessage>;
using ResponseMessage = FBS_TypeBufferOwned<message::ResponseMessage>;

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

/**
 * Table for storing Tensor metadata on rank zero. This is used for error checking and size calculation, as well as
 * determining when a reduction is ready to be done (when all nodes are ready to do it).
 */
using MessageTable       = std::unordered_map<std::string, std::vector<RequestMessage>>;
using MessageTableCloned = std::unordered_map<std::string, std::vector<RequestMessageCloned>>;

/**
 * Store the RequestMessage for a name and return whether the total count of RequestMessages for that tensor is now
 * equal to the MPI size (and thus we are ready to reduce the tensor).
 */
bool IncreTensorCount(MessageTable& table, RequestMessage&& msg, int mpi_size);

ResponseMessage ConstructResponseMessage(const MessageTable& table, const std::string& name) {
  auto it = table.find(name);
  CHECK(it != table.end()) << "message table doesn't have item called " << name;

  const auto& requests = it->second;
  CHECK_GT(requests.size(), 0);

  bool error = false;
  std::stringstream error_stream;

  // Check that all data types are identical
  auto data_type = requests[0].msg().tensor_type();
  for (int i = 1; i < requests.size() && !error; i++) {
    auto type = requests[i].msg().tensor_type();
    if (data_type != type) {
      error = true;
      error_stream << "Mismatch data types found: " << data_type << " vs " << type << ".";
    }
  }

  // Check that all requested operations are the same.
  auto operation_type = requests[0].msg().request_type();
  for (int i = 1; i < requests.size() && !error; i++) {
    auto type = requests[0].msg().request_type();
    if (operation_type != type) {
      error = true;
      error_stream << "Mismatched operations found: " << operation_type << " vs " << type << ".";
    }
  }

  // If we are doing an allreduce, check that all the tensor shape are identical.

  // TODO
}

template <typename FBS_T>
void FBS_TypeBufferOwned<FBS_T>::Copy(const uint8_t* buffer, size_t len) {
  CHECK(!buffer_) << "FBS_TypeBufferOwned duplicate assign found";
  len_    = len;
  buffer_ = reinterpret_cast<uint8_t*>(std::malloc(len));
  CHECK(buffer_) << "allocate memory failed";
  std::memcpy(buffer_, buffer, len);
}

}  // namespace collective
}  // namespace tips
