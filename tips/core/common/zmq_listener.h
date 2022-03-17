#pragma once
#include <zmq.h>

#include <mutex>
#include <thread>
#include <vector>

#include "tips/core/common/logging.h"
#include "tips/core/common/thread_pool.h"
#include "tips/core/utils/spinlock.h"

namespace tips {

void zmq_bind_random_port(const std::string& ip, void* socket, std::string& addr, int& port);

/**
 * ZmqListener: a listen service based on ZMQ.
 */
class ZmqListener {
 public:
  ZmqListener() = default;

  explicit ZmqListener(void* zmq_ctx) { SetZmqCtx(zmq_ctx); }

  ~ZmqListener();

  virtual void MainLoop()        = 0;
  virtual bool ServiceComplete() = 0;

  void SetThreadNum(int num) {
    CHECK_GT(num, 0);
    thread_pool_.ResizeThreads(num);
  }

  int thread_num() const { return thread_pool_.size(); }

  void StartService() {
    CHECK_GT(thread_num(), 0);
    thread_pool_.Run([this](int tid) { MainLoop(); });
  }

  void StopService() { thread_pool_.Stop(); }

  void SetZmqCtx(void* ctx) {
    zmq_ctx_ = ctx;
    CHECK(receiver_ = zmq_socket(zmq_ctx_, ZMQ_PULL));
  }

  void SetRecvIp(const std::string& ip) {
    CHECK(!ip.empty());
    recv_ip_ = ip;
  }

  int Listen();

 private:
  void* zmq_ctx_{};
  void* receiver_{};
  std::string recv_addr_;
  std::string recv_ip_;
  int recv_port_{-1};
  // listen service
  ThreadPool thread_pool_;
};

}  // namespace tips
