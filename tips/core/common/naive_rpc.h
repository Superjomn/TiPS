#pragma once
#include <glog/logging.h>
#include <mpi.h>
#include <zmq.h>

#include <flatbuffers/flatbuffers.h>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_set>

#include "tips/core/common/common.h"
#include "tips/core/common/macro.h"
#include "tips/core/common/zmq_message.h"
#include "tips/core/mpi/tips_mpi.h"

// This file implements a naive RPC.

namespace tips {

class RpcService;
class RpcRequest;
class BinaryArchive {};

#define RPC_MESSAGE_TYPE_FOREACH(op__) op__(REQUEST) op__(RESPONSE)

enum class RpcMsgType : int {
#define __(ITEM) ITEM,
  RPC_MESSAGE_TYPE_FOREACH(__)
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

using RpcCallback  = std::function<void(const RpcMsgHead&, NaiveBuffer&)>;
using RpcCallback2 = std::function<void(const RpcMsgHead&, uint8_t* buffer)>;

/**
 * RpcService represents a service in the RPC framework. The callback will be invoked when a Request arrive.
 */
class RpcService {
 public:
  explicit RpcService(RpcCallback2 callback);

  ~RpcService() {
    MPI_Barrier(mpi_comm());
    CHECK_EQ(request_counter_, 0);
  }

  /**
   * Get the service address belong to the \p rank -th node.
   */
  RpcService* remote_service(size_t rank);

  RpcCallback2& callback() { return callback_; }

  SWIFTS_DISALLOW_COPY_AND_ASSIGN(RpcService)

  friend class RpcServer;

 private:
  inline void IncRequest() { ++request_counter_; }
  inline void DecRequest() { --request_counter_; }

 private:
  RpcCallback2 callback_;

  std::vector<RpcService*> remote_service_ptrs_;

  int request_counter_{};
};

class RpcRequest {
 public:
  explicit RpcRequest(RpcCallback2 callback) : callback_(callback) {}

  RpcCallback2& callback() { return callback_; }

 private:
  RpcCallback2 callback_;
};

/**
 * An PRC server.
 *
 * Usage:
 *
 * RpcServer rpc;
 * auto* service = rpc.AddService([](RpcMsgHead head, NaiveBuffer& buffer) {
 *   LOG(INFO) << "get a message";
 *   });
 *
 * NaiveBuffer buf;
 * buf << 1;
 * rpc.SendRequest(1, service, buf);
 */
class RpcServer {
 public:
  RpcServer() = default;
  RpcServer(int num_connection, int num_listen_threads, int zmq_num_threads)
      : num_connection_(num_connection), num_listen_threads_(num_listen_threads), zmq_num_threads_(zmq_num_threads) {}

  RpcService* AddService(RpcCallback callback);
  RpcService* AddService(RpcCallback2 callback);

  //! Initialize the server run loop.
  void Initialize();

  //! Finalize will force the server quit.
  void Finalize();

  void SendRequest(int server_id, RpcService* service, const NaiveBuffer& buf, RpcCallback callback);

  void SendRequest(int server_id,
                   RpcService* service,
                   const flatbuffers::FlatBufferBuilder& builder,
                   RpcCallback2 callback);

  void SendResponse(RpcMsgHead head, const NaiveBuffer& buf);

  void SendResponse(RpcMsgHead head, const flatbuffers::FlatBufferBuilder& builder);

  ~RpcServer();

  SWIFTS_DISALLOW_COPY_AND_ASSIGN(RpcServer)

 private:
  int BindRandomPort();

  void StartRunLoop();

  std::unique_ptr<ZmqMessage> MakeMessage(const RpcMsgHead& head, const NaiveBuffer& buf);
  std::unique_ptr<ZmqMessage> MakeMessage(const RpcMsgHead& head, const flatbuffers::FlatBufferBuilder& builder);

 private:
  int num_connection_{1};
  int num_listen_threads_{1};
  int zmq_num_threads_{1};

  void* zmq_ctx_{};

  //! Receiver zmq socket.
  void* receiver_{};
  std::mutex recv_mutex_;

  //! Senders' smq sockets.
  std::vector<void*> senders_;
  std::vector<std::mutex> sender_mutexs_;

  std::vector<std::thread> listen_threads_;

  std::unordered_set<RpcService*> services_;
};

}  // namespace tips
