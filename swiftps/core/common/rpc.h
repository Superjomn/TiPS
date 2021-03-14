#pragma once
#include <mpi.h>
#include <zmq.h>

#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_set>

#include "swiftps/core/common/common.h"
#include "swiftps/core/common/macro.h"
#include "swiftps/core/mpi/mpi.h"

namespace swifts {

class RpcService;
class RpcRequest;
class BinaryArchive {};

struct RpcMsgHead {
  RpcService* service{};
  RpcRequest* request{};

  int client_id{-1};
  int server_id{-1};

  enum class MessageType : int { REQUEST, RESPONSE };

  MessageType message_type;
};

using RpcCallback = std::function<void(const RpcMsgHead&, BinaryArchive&)>;

class RpcService {
 public:
  RpcService() = default;

  explicit RpcService(RpcCallback&& callback) : callback_(callback) {
    remote_service_ptrs_.resize(mpi_size(), nullptr);
    RpcService* my_ptr = this;
    MPI_Allgather(&my_ptr, 1, MPI_LONG_LONG, &remote_service_ptrs_[0], 1, MPI_LONG_LONG, mpi_comm());
    MPI_Barrier(mpi_comm());
  }

  ~RpcService() {
    MPI_Barrier(mpi_comm());
    CHECK_EQ(request_counter_, 0);
  }

  RpcService* remote_addr(int rank) { return remote_service_ptrs_[rank]; }

  RpcCallback& callback() { return callback_; }

  void IncRequest() { ++request_counter_; }
  void DescRequest() { --request_counter_; }

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
  RpcService* AddService(RpcCallback callback);

  void Initialize() {
    MPI_Barrier(mpi_comm());
    CHECK(!zmq_ctx_) << "Duplicate initialization found";
    CHECK(zmq_ctx_ = zmq_ctx_new());
    CHECK_EQ(zmq_ctx_set(zmq_ctx_, ZMQ_IO_THREADS, zmq_num_threads_), 0);

    CHECK(receiver_ = zmq_socket(zmq_ctx_, ZMQ_PULL));
    CHECK(zmq_setsockopt(receiver_, ZMQ_RCVHWM, &*std::make_unique<int>(0), sizeof(int)));
    CHECK(zmq_setsockopt(receiver_, ZMQ_BACKLOG, &*std::make_unique<int>(3000), sizeof(int)));
    senders_.resize(mpi_size());

    for (int i = 0; i < mpi_size(); i++) {
      CHECK(senders_[i] = zmq_socket(zmq_ctx_, ZMQ_PUSH));
      CHECK_EQ(zmq_setsockopt(senders_[i], ZMQ_SNDHWM, &*std::make_unique<int>(0), sizeof(int)), 0);
    }
    std::vector<std::mutex> tmp_muts(mpi_size());
    sender_mutexs_.swap(tmp_muts);

    for (int conn = 0; conn < num_connection_; conn++) {
      std::vector<int> ports(mpi_size());
    }
  }

  ~RpcServer();

  SWIFTS_DISALLOW_COPY_AND_ASSIGN(RpcServer)

 private:
  int BindRandomPort() {
    for (;;) {
      int port         = 1024 + rand() % (65536 - 1024);
      std::string addr = StringFormat("tcp://%s:%d", mpi_ip().c_str(), port);
      int res          = 0;
      PCHECK((res = zmq_bind(receiver_, addr.c_str()), res == 0 || errno == EADDRINUSE));

      if (res == 0) {
        return port;
      }
    }
  }

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
