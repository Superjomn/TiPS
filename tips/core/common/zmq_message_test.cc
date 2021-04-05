#include "tips/core/common/zmq_message.h"

#include <gtest/gtest.h>

#include <string>

namespace tips {

TEST(ZmqMessage, Move) {
  ZmqMessage message;
  std::string info = "hello";
  message.Resize(info.size());
  memcpy(message.buffer(), info.data(), info.size());

  ZmqMessage message1(std::move(message));
  ASSERT_EQ(message1.length(), info.size());
  ASSERT_EQ(memcmp(info.data(), message1.buffer(), info.size()), 0);

  ASSERT_EQ(message.length(), 0);
}

}  // namespace tips
