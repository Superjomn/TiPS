#include "tips/core/common/naive_rpc.h"

namespace tips {

RpcServer::~RpcServer() {
  for (auto &item : services_) {
    if (item.second) delete item.second;
  }
}

RpcService *RpcServer::AddService(const std::string &type, RpcCallback callback) {
  auto *new_service = new RpcService(std::move(callback));
  CHECK(!services_.count(type)) << "duplicate add service [" << type << "]";
  services_.emplace(type, new_service);
  return new_service;
}

void RpcServer::StartRunLoop() {
  while (true) {
    ZmqMessage msg;

    // Receive a message
    if (num_listen_threads_ > 1) {
      recv_mutex_.lock();
    }
    CHECK_GE(ignore_signal_call(zmq_msg_recv, msg.zmq_msg(), receiver_, 0), 0);
    if (num_listen_threads_ > 1) {
      recv_mutex_.unlock();
    }

    // Read the message

    if (msg.length() == 0) {  // Terminate
      LOG(WARNING) << "Receive a terminate message, RPC server to quit";
      return;
    }

    if (msg.length() < sizeof(RpcMsgHead)) {
      LOG(ERROR) << "Unknown message. Received data size " << msg.length() << ", message ignored";
      continue;
    }

    std::vector<char> buffer;
    buffer.resize(msg.length());

    std::memcpy(buffer.data(), msg.buffer(), msg.length());
    msg.Release();

    // Parse the message content.
    RpcMsgHead *head = reinterpret_cast<RpcMsgHead *>(buffer.data());

    uint8_t *data = reinterpret_cast<uint8_t *>(buffer.data() + sizeof(RpcMsgHead));
    switch (head->message_type) {
      case RpcMsgType::REQUEST: {
        CHECK_EQ(head->server_id, mpi_rank());
        head->service->callback()(*head, data);
      } break;

      case RpcMsgType::RESPONSE: {
        CHECK_EQ(head->client_id, mpi_rank());
        VLOG(3) << "call response callback...";
        head->request->callback()(*head, data);
        VLOG(3) << "done call response callback...";
        CHECK(head->request);
        delete head->request;
        CHECK(head->service);
        head->service->DecRequest();
      } break;

      default:
        LOG(FATAL) << "Unknown message type found";
    }
  }
}

std::unique_ptr<ZmqMessage> RpcServer::MakeMessage(const RpcMsgHead &head, const FlatBufferBuilder &buf) {
  CHECK_NE(head.server_id, -1);
  CHECK_NE(head.client_id, -1);
  CHECK(head.service);
  CHECK(head.request);

  size_t len = sizeof(head);
  len += buf.GetSize();

  auto msg = std::make_unique<ZmqMessage>();
  msg->Resize(len);
  len = 0;

  std::memcpy(msg->buffer() + len, &head, sizeof(head));
  len += sizeof(head);

  std::memcpy(msg->buffer() + len, buf.GetBufferPointer(), buf.GetSize());

  return msg;
}

void RpcServer::SendResponse(RpcMsgHead head, const FlatBufferBuilder &buf) {
  CHECK(initialized_) << "Server should be initialized first";
  CHECK(!finalized_) << "Server is finailized";

  CHECK_EQ(head.server_id, mpi_rank());
  CHECK_GE(head.client_id, 0);
  CHECK_LT(head.client_id, mpi_size());

  VLOG(2) << mpi_rank() << " to send response...";

  head.service      = head.service->remote_service(head.client_id);
  head.message_type = RpcMsgType::RESPONSE;

  VLOG(3) << mpi_rank() << " to send response";
  VLOG(3) << "- server_id " << head.server_id;
  VLOG(3) << "- client_id " << head.client_id;
  VLOG(3) << "- service " << head.service;
  VLOG(3) << "--------";

  auto msg = MakeMessage(head, buf);
  sender_mutexes_[head.client_id].lock();
  CHECK_GE(ignore_signal_call(zmq_msg_send, msg->zmq_msg(), senders_[head.client_id], 0), 0);
  sender_mutexes_[head.client_id].unlock();
}

void RpcServer::SendRequest(int server_id, RpcService *service, const FlatBufferBuilder &buf, RpcCallback callback) {
  CHECK(initialized_) << "Server should be initialized first";
  CHECK(!finalized_) << "Server is finailized";

  CHECK_GE(server_id, 0);
  CHECK_LT(server_id, mpi_size());
  CHECK(service);
  CHECK(callback);

  service->IncRequest();

  RpcMsgHead head;
  head.service      = service->remote_service(server_id);  // destination
  head.request      = new RpcRequest(std::move(callback));
  head.client_id    = mpi_rank();
  head.server_id    = server_id;
  head.message_type = RpcMsgType::REQUEST;

  VLOG(4) << "to send request.service " << head.service;
  auto msg = MakeMessage(head, buf);
  sender_mutexes_[server_id].lock();
  CHECK_GE(ignore_signal_call(zmq_msg_send, msg->zmq_msg(), senders_[server_id], 0), 0);
  sender_mutexes_[server_id].unlock();
}

void RpcServer::Finalize() {
  CHECK(initialized_);
  CHECK(!finalized_) << "Duplicate finalization found";

  // Update state.
  finalized_ = true;

  VLOG(1) << "#### to finalize";
  CHECK(zmq_ctx_);

  VLOG(3) << "tell all the threads to quit";
  for (int i = 0; i < num_listen_threads_; i++) {
    sender_mutexes_[mpi_rank()].lock();
    CHECK_GE(ignore_signal_call(zmq_msg_send, ZmqMessage().zmq_msg(), senders_[mpi_rank()], 0), 0);
    sender_mutexes_[mpi_rank()].unlock();
  }

  for (int i = 0; i < num_listen_threads_; i++) {
    if (listen_threads_[i].joinable()) {
      listen_threads_[i].join();
    }
  }

  for (int i = 0; i < mpi_size(); i++) {
    ZCHECK(zmq_close(senders_[i]));
  }

  ZCHECK(zmq_close(receiver_));
  ZCHECK(zmq_ctx_destroy(zmq_ctx_));

  mpi_barrier();

  for (auto &item : services_) {
    delete item.second;
    item.second = nullptr;
  }
}

void RpcServer::Initialize() {
  // Update state.
  initialized_ = true;

  MPI_Barrier(mpi_comm());
  CHECK(!zmq_ctx_) << "Duplicate initialization found";
  CHECK(zmq_ctx_ = zmq_ctx_new());
  CHECK_EQ(zmq_ctx_set(zmq_ctx_, ZMQ_IO_THREADS, zmq_num_threads_), 0);

  CHECK(receiver_ = zmq_socket(zmq_ctx_, ZMQ_PULL));

  int v0 = 0;
  int v1 = 3000;
  ZCHECK(zmq_setsockopt(receiver_, ZMQ_RCVHWM, &v0, sizeof(int)));
  ZCHECK(zmq_setsockopt(receiver_, ZMQ_BACKLOG, &v1, sizeof(int)));

  senders_.resize(mpi_size());

  for (int i = 0; i < mpi_size(); i++) {
    CHECK(senders_[i] = zmq_socket(zmq_ctx_, ZMQ_PUSH));
    int v0 = 0;
    CHECK_EQ(zmq_setsockopt(senders_[i], ZMQ_SNDHWM, &v0, sizeof(int)), 0);
  }
  std::vector<std::mutex> tmp_muts(mpi_size());
  sender_mutexes_.swap(tmp_muts);

  for (int conn = 0; conn < num_connection_; conn++) {
    std::vector<int> ports(mpi_size());
    ports[mpi_rank()] = BindRandomPort();
    // No need to check port duplication, the ZMQ connect will fail if duplcate.
    CHECK_EQ(MPI_Allgather(MPI_IN_PLACE, 0, MPI_INT, &ports[0], 1, MPI_INT, mpi_comm()), 0);

    for (int i = 0; i < mpi_size(); i++) {
      LOG(INFO) << "ip " << i << " " << MpiContext::Global().ip(i);
      CHECK_EQ(ignore_signal_call(zmq_connect,
                                  senders_[i],
                                  StringFormat("tcp://%s:%d", MpiContext::Global().ip(i).c_str(), ports[i]).c_str()),
               0);
    }

    for (int i = 0; i < num_listen_threads_; i++) {
      listen_threads_.emplace_back([this] { StartRunLoop(); });
    }

    mpi_barrier();
  }
}

int RpcServer::BindRandomPort() {
  for (;;) {
    int port         = 1024 + rand() % (65536 - 1024);
    std::string addr = StringFormat("tcp://%s:%d", MpiContext::Global().ip().c_str(), port);
    int res          = 0;
    PCHECK((res = zmq_bind(receiver_, addr.c_str()), res == 0 || errno == EADDRINUSE));

    if (res == 0) {
      return port;
    }
  }
}

const char *GetRpcMsgTypeRepr(RpcMsgType type) {
  switch (type) {
#define __(item__)           \
  case (RpcMsgType::item__): \
    return #item__;          \
    break;

    RPC_MESSAGE_TYPE_FOREACH(__)
#undef __
  }
  return nullptr;
}

std::ostream &operator<<(std::ostream &os, RpcMsgType type) {
  os << GetRpcMsgTypeRepr(type);
  return os;
}

RpcService::RpcService(RpcCallback callback) : callback_(std::move(callback)) {
  remote_service_ptrs_.resize(mpi_size(), nullptr);
  RpcService *my_ptr = this;
  CHECK_EQ(sizeof(void *), sizeof(long long));
  MPI_Allgather(&my_ptr, 1, MPI_LONG_LONG, &remote_service_ptrs_[0], 1, MPI_LONG_LONG, mpi_comm());
  mpi_barrier();

  if (mpi_rank() == 0) {
    for (int i = 0; i < mpi_size(); i++) {
      LOG(INFO) << i << "-service: " << remote_service_ptrs_[i];
    }
  }
}

RpcService *RpcService::remote_service(size_t rank) {
  CHECK_LT(rank, remote_service_ptrs_.size());
  return remote_service_ptrs_[rank];
}

RpcService *RpcServer::LookupService(const std::string &type) const {
  auto it = services_.find(type);
  return it == services_.end() ? nullptr : it->second;
}

RpcServer &RpcServer::Global() {
  // NOTE One should manually call RpcServer::Initialize and RpcServer::Finalize before and after using this singleton.
  static RpcServer x;
  return x;
}

}  // namespace tips
