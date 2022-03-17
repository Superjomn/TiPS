#include "tips/core/common/zmq_listener.h"

namespace tips {

void zmq_bind_random_port(const std::string &ip, void *socket, std::string &addr, int &port) {
  for (;;) {
    addr = "";
    port = 1024 + rand() % (65536 - 1024);
    StringFormat(addr, "tcp://%s:%d", ip.c_str(), port);
    // ATTENTION: fix the wied memory leak
    // add the LOG valhind detect no memory leak, else ...
    LOG(WARNING) << "try addr: " << addr;
    int res = 0;
    CHECK((res = zmq_bind(socket, addr.c_str()),
           res == 0 || errno == EADDRINUSE));  // port is already in use
    if (res == 0) break;
  }
}

ZmqListener::~ZmqListener() {
  if (receiver_) {
    CHECK(0 == zmq_close(receiver_));
    receiver_ = nullptr;
  }
}

int ZmqListener::Listen() {
  if (recv_ip_.empty()) {
    recv_ip_ = GetLocalIp();
  }
  CHECK_EQ(recv_port_, -1) << "local receiver can only listen once";
  zmq_bind_random_port(recv_ip_, receiver_, recv_addr_, recv_port_);
  LOG(INFO) << "client listen to address:\t" << recv_addr_;
  return recv_port_;
}
}  // namespace tips