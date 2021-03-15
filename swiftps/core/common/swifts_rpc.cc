#include "swiftps/core/common/swifts_rpc.h"

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
    if (msg.length() >= sizeof(*msg.zmq_msg())) {
      in_buf.LoadFromMemory(msg.buffer(), msg.length());
      msg.Release();
    } else {
      in_buf.Require(msg.length());
      memcpy(in_buf.data(), msg.buffer(), msg.length());
      in_buf.Consume(msg.length());
    }

    // Parse the message content.
    NaiveBuffer read_buf(in_buf.data(), in_buf.size());  // TODO(Superjomn) support zero copy
    RpcMsgHead *head = reinterpret_cast<RpcMsgHead *>(read_buf.cursor());
    read_buf.Consume(sizeof(RpcMsgHead));

    LOG(INFO) << mpi_rank() << " .. get message.service " << head->service;

    switch (head->message_type) {
      case RpcMsgType::REQUEST: {
        CHECK_EQ(head->server_id, mpi_rank());
        LOG(INFO) << mpi_rank() << " get request message.service " << head->service;
        head->service->callback()(*head, read_buf);
      } break;

      case RpcMsgType::RESPONSE: {
        LOG(INFO) << mpi_rank() << " get response message service " << head->service;
        CHECK_EQ(head->client_id, mpi_rank());
        LOG(INFO) << "call response callback...";
        head->request->callback()(*head, read_buf);
        LOG(INFO) << "done call response callback...";
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

std::unique_ptr<ZmqMessage> RpcServer::MakeMessage(const RpcMsgHead &head, NaiveBuffer *bufs, size_t n) {
  CHECK_NE(head.server_id, -1);
  CHECK_NE(head.client_id, -1);
  CHECK(head.service);
  CHECK(head.request);

  LOG(INFO) << "Making message: ";
  LOG(INFO) << "msg.service: " << head.service;
  LOG(INFO) << "------";
  size_t len = sizeof(head);
  for (size_t i = 0; i < n; i++) {
    len += bufs[i].size();
  }

  auto msg = std::make_unique<ZmqMessage>();
  msg->Resize(len);
  len = 0;

  memcpy(msg->buffer() + len, &head, sizeof(head));
  len += sizeof(head);

  for (size_t i = 0; i < n; i++) {
    // CHECK_EQ(static_cast<void *>(bufs[i].data()), static_cast<void *>(bufs[i].cursor()));
    memcpy(msg->buffer() + len, bufs[i].data(), bufs[i].size());
    len += bufs[i].size();
  }

  return msg;
}

void RpcServer::SendResponse(RpcMsgHead head, NaiveBuffer *bufs, int n) {
  CHECK_EQ(head.server_id, mpi_rank());
  CHECK_GE(head.client_id, 0);
  CHECK_LT(head.client_id, mpi_size());

  LOG(INFO) << mpi_rank() << " to send response...";

  head.service      = head.service->remote_service(head.client_id);
  head.message_type = RpcMsgType::RESPONSE;

  LOG(INFO) << mpi_rank() << " to send response";
  LOG(INFO) << "- server_id " << head.server_id;
  LOG(INFO) << "- client_id " << head.client_id;
  LOG(INFO) << "- service " << head.service;
  LOG(INFO) << "--------";

  auto msg = MakeMessage(head, bufs, n);
  sender_mutexs_[head.client_id].lock();
  CHECK_GE(ignore_signal_call(zmq_msg_send, msg->zmq_msg(), senders_[head.client_id], 0), 0);
  sender_mutexs_[head.client_id].unlock();
  LOG(INFO) << mpi_rank() << " finish send response";
}

void RpcServer::SendRequest(int server_id, RpcService *service, NaiveBuffer *bufs, int n, RpcCallback callback) {
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

  LOG(INFO) << "to send request.service " << head.service;
  auto msg = MakeMessage(head, bufs, n);
  sender_mutexs_[server_id].lock();
  CHECK_GE(ignore_signal_call(zmq_msg_send, msg->zmq_msg(), senders_[server_id], 0), 0);
  sender_mutexs_[server_id].unlock();
}

void RpcServer::Finalize() {
  LOG(INFO) << "#### to finalize";
  CHECK(zmq_ctx_);

  LOG(INFO) << "tell all the threads to quit";
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

  LOG(INFO) << "zmq_setsockopt ...";
  int v0 = 0;
  int v1 = 3000;
  ZCHECK(zmq_setsockopt(receiver_, ZMQ_RCVHWM, &v0, sizeof(int)));
  ZCHECK(zmq_setsockopt(receiver_, ZMQ_BACKLOG, &v1, sizeof(int)));

  LOG(INFO) << "Initialize senders";
  senders_.resize(mpi_size());

  for (int i = 0; i < mpi_size(); i++) {
    CHECK(senders_[i] = zmq_socket(zmq_ctx_, ZMQ_PUSH));
    int v0 = 0;
    CHECK_EQ(zmq_setsockopt(senders_[i], ZMQ_SNDHWM, &v0, sizeof(int)), 0);
  }
  std::vector<std::mutex> tmp_muts(mpi_size());
  sender_mutexs_.swap(tmp_muts);

  LOG(INFO) << "To connect send socket...";
  for (int conn = 0; conn < num_connection_; conn++) {
    std::vector<int> ports(mpi_size());
    LOG(INFO) << "to assign random port...";
    ports[mpi_rank()] = BindRandomPort();
    LOG(INFO) << "get random port: " << ports[mpi_rank()];
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
    int port = 1024 + rand() % (65536 - 1024);
    LOG(INFO) << "try port: " << port;
    std::string addr = StringFormat("tcp://%s:%d", MpiContext::Global().ip().c_str(), port);
    LOG(INFO) << "try addr: " << addr;
    int res = 0;
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

}  // namespace swifts
