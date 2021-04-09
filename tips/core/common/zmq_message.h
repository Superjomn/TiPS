#pragma once
#include <zmq.h>

#include <memory>

#include "tips/core/common/common.h"

namespace tips {

struct RpcMsgHead;

class ZmqMessage : public std::enable_shared_from_this<ZmqMessage> {
 public:
  ZmqMessage() { ZCHECK(zmq_msg_init(&data_)); }

  ZmqMessage(ZmqMessage&& other) {
    data_ = other.data_;
    ZCHECK(zmq_msg_init(&other.data_));
  }

  ZmqMessage(char* buf, size_t size) {
    ZCHECK(zmq_msg_init_size(&data_, size));
    memcpy(buffer(), buf, size);
  }

  ZmqMessage& operator=(ZmqMessage&& other) {
    data_ = other.data_;
    ZCHECK(zmq_msg_init(&other.data_));
    return *this;
  }

  void Reset() {
    ZCHECK(zmq_msg_close(&data_));
    ZCHECK(zmq_msg_init(&data_));
  }

  void Resize(size_t size) {
    ZCHECK(zmq_msg_close(&data_));
    ZCHECK(zmq_msg_init_size(&data_, size));
  }

  void Assign(char* buf, size_t size) {
    Resize(size);
    memcpy(buffer(), buf, size);
  }

  void Release() { ZCHECK(zmq_msg_init(&data_)); }

  std::shared_ptr<ZmqMessage> getptr() { return shared_from_this(); }

  ~ZmqMessage() { zmq_msg_close(&data_); }

  void* buffer() { return zmq_msg_data(&data_); }
  const void* buffer() const { return static_cast<const void*>(zmq_msg_data(&data_)); }

  size_t length() const { return zmq_msg_size(&data_); }

  zmq_msg_t* zmq_msg() { return &data_; }

  TIPS_DISALLOW_COPY_AND_ASSIGN(ZmqMessage)

 private:
  mutable zmq_msg_t data_;
};

const RpcMsgHead* GetMsgHead(const ZmqMessage& msg);
const void* GetMsgContent(const ZmqMessage& msg);

}  // namespace tips
