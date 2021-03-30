#pragma once
#include <zmq.h>

#include "tips/core/common/common.h"

namespace tips {

class ZmqMessage {
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

  char* buffer() { return static_cast<char*>(zmq_msg_data(&data_)); }
  size_t length() const { return zmq_msg_size(&data_); }

  zmq_msg_t* zmq_msg() { return &data_; }

  TIPS_DISALLOW_COPY_AND_ASSIGN(ZmqMessage)

 private:
  zmq_msg_t data_;
};

}  // namespace tips
