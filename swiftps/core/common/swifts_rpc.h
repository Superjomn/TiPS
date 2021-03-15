#pragma once
#include <glog/logging.h>
#include <mpi.h>
#include <zmq.h>

#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_set>

#include "swiftps/core/common/common.h"
#include "swiftps/core/common/macro.h"
#include "swiftps/core/common/zmq_message.h"
#include "swiftps/core/mpi/swifts_mpi.h"

namespace swifts {

class RpcService;
class RpcRequest;
class BinaryArchive {};

#define MESSAGE_TYPE_FOREACH(op__) op__(REQUEST) op__(RESPONSE)

enum class RpcMsgType : int {
#define __(ITEM) ITEM,
  MESSAGE_TYPE_FOREACH(__)
#undef __
};

const char* GetRpcMsgTypeRepr(RpcMsgType type);

std::ostream& operator<<(std::ostream& os, RpcMsgType type);

struct RpcMsgHead {
  RpcService* service{};
  RpcRequest* request{};

  int client_id{-1};
  int server_id{-1};

  RpcMsgType message_type;
};

using RpcCallback = std::function<void(const RpcMsgHead&, NaiveBuffer&)>;

class RpcService {
 public:
  explicit RpcService(RpcCallback callback) : callback_(std::move(callback)) {
    remote_service_ptrs_.resize(mpi_size(), nullptr);
    RpcService* my_ptr = this;
    CHECK_EQ(sizeof(void*), sizeof(long long));
    MPI_Allgather(&my_ptr, 1, MPI_LONG_LONG, &remote_service_ptrs_[0], 1, MPI_LONG_LONG, mpi_comm());
    mpi_barrier();

    if (mpi_rank() == 0) {
      for (int i = 0; i < mpi_size(); i++) {
        LOG(INFO) << i << "-service: " << remote_service_ptrs_[i];
      }
    }
  }

  ~RpcService() {
    MPI_Barrier(mpi_comm());
    CHECK_EQ(request_counter_, 0);
  }

  RpcService* remote_service(size_t rank) {
    CHECK_LT(rank, remote_service_ptrs_.size());
    return remote_service_ptrs_[rank];
  }

  RpcCallback& callback() { return callback_; }

  void IncRequest() { ++request_counter_; }
  void DecRequest() { --request_counter_; }

  SWIFTS_DISALLOW_COPY_AND_ASSIGN(RpcService)

 private:
  RpcCallback callback_;

  std::vector<RpcService*> remote_service_ptrs_;

  int request_counter_{};
};

class RpcRequest {
 public:
  explicit RpcRequest(RpcCallback callback) : callback_(callback) {}

  RpcCallback& callback() { return callback_; }

 private:
  RpcCallback callback_;
};

class RpcServer {
 public:
  RpcServer() = default;

  RpcService* AddService(RpcCallback callback);

  void Initialize();

  void Finalize();

  void SendRequest(int server_id, RpcService* service, NaiveBuffer* bufs, int n, RpcCallback callback);

  void SendResponse(RpcMsgHead head, NaiveBuffer* bufs, int n);

  ~RpcServer();

  SWIFTS_DISALLOW_COPY_AND_ASSIGN(RpcServer)

 private:
  int BindRandomPort();
  void Run();
  std::unique_ptr<ZmqMessage> MakeMessage(const RpcMsgHead& head, NaiveBuffer* bufs, size_t n);

 private:
  int num_connection_{1};
  int num_threads_{1};
  int zmq_num_threads_{1};

  void* zmq_ctx_{};

  void* receiver_{};
  std::mutex recv_mutex_;

  std::vector<void*> senders_;
  std::vector<std::mutex> sender_mutexs_;

  std::vector<std::thread> threads_;

  std::unordered_set<RpcService*> services_;
};

}  // namespace swifts
