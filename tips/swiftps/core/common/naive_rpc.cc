#include "swiftps/core/common/naive_rpc.h"

namespace swifts {

RpcServer::~RpcServer() {
  for (auto *x : services_) {
    delete x;
  }
}

RpcService *RpcServer::AddService(RpcCallback callback) {
  auto *new_service = new RpcService(std::move(callback));
  services_.insert(new_service);
  return new_service;
}

void RpcServer::Run() {
  while (true) {
    ZmqMessage msg;

    // Receive a message
    if (num_threads_ > 1) {
      recv_mutex_.lock();
    }
    CHECK_GE(ignore_signal_call(zmq_msg_recv, msg.zmq_msg(), receiver_, 0), 0);
    if (num_threads_ > 1) {
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

    NaiveBuffer in_buf;
    in_buf.LoadFromMemory(msg.buffer(), msg.length());
    msg.Release();

    // Parse the message content.
    NaiveBuffer read_buf(in_buf.data(), in_buf.size());  // TODO(Superjomn) support zero copy
    CHECK_EQ(read_buf.data(), read_buf.cursor());
    RpcMsgHead *head = reinterpret_cast<RpcMsgHead *>(read_buf.cursor());
    read_buf.Consume(sizeof(RpcMsgHead));

    switch (head->message_type) {
      case RpcMsgType::REQUEST: {
        CHECK_EQ(head->server_id, mpi_rank());
        head->service->callback()(*head, read_buf);
      } break;

      case RpcMsgType::RESPONSE: {
        CHECK_EQ(head->client_id, mpi_rank());
        VLOG(3) << "call response callback...";
        head->request->callback()(*head, read_buf);
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

std::unique_ptr<ZmqMessage> RpcServer::MakeMessage(const RpcMsgHead &head, const NaiveBuffer &buf) {
  CHECK_NE(head.server_id, -1);
  CHECK_NE(head.client_id, -1);
  CHECK(head.service);
  CHECK(head.request);

  size_t len = sizeof(head);
  len += buf.size();

  auto msg = std::make_unique<ZmqMessage>();
  msg->Resize(len);
  len = 0;

  memcpy(msg->buffer() + len, &head, sizeof(head));
  len += sizeof(head);

  memcpy(msg->buffer() + len, buf.data(), buf.size());

  return msg;
}

void RpcServer::SendResponse(RpcMsgHead head, const NaiveBuffer &buf) {
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
  sender_mutexs_[head.client_id].lock();
  CHECK_GE(ignore_signal_call(zmq_msg_send, msg->zmq_msg(), senders_[head.client_id], 0), 0);
  sender_mutexs_[head.client_id].unlock();
}

void RpcServer::SendRequest(int server_id, RpcService *service, const NaiveBuffer &buf, RpcCallback callback) {
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
  sender_mutexs_[server_id].lock();
  CHECK_GE(ignore_signal_call(zmq_msg_send, msg->zmq_msg(), senders_[server_id], 0), 0);
  sender_mutexs_[server_id].unlock();
}

void RpcServer::Finalize() {
  VLOG(1) << "#### to finalize";
  CHECK(zmq_ctx_);

  VLOG(3) << "tell all the threads to quit";
  for (int i = 0; i < num_threads_; i++) {
    sender_mutexs_[mpi_rank()].lock();
    CHECK_GE(ignore_signal_call(zmq_msg_send, ZmqMessage().zmq_msg(), senders_[mpi_rank()], 0), 0);
    sender_mutexs_[mpi_rank()].unlock();
  }

  for (int i = 0; i < num_threads_; i++) {
    if (threads_[i].joinable()) {
      threads_[i].join();
    }
  }

  for (int i = 0; i < mpi_size(); i++) {
    ZCHECK(zmq_close(senders_[i]));
  }

  ZCHECK(zmq_close(receiver_));
  ZCHECK(zmq_ctx_destroy(zmq_ctx_));

  mpi_barrier();
}

void RpcServer::Initialize() {
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
  sender_mutexs_.swap(tmp_muts);

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

    for (int i = 0; i < num_threads_; i++) {
      threads_.emplace_back([this] { Run(); });
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

    MESSAGE_TYPE_FOREACH(__)
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
}  // namespace swifts
