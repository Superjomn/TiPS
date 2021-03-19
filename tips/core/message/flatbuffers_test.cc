#include <glog/logging.h>
#include <gtest/gtest.h>
#include <boost/timer.hpp>
#include "tips/core/common/naive_buffer.h"
#include "tips/core/message/test0_generated.h"

namespace tips {

using namespace tips::test::test_message0;
using namespace flatbuffers;
TEST(flatbuffers, basic) {
  auto fn0 = [] {
    FlatBufferBuilder builder;
    auto hello = builder.CreateString("hello world");

    MessageRequestBuilder message(builder);
    message.add_greet(hello);
    message.add_v(13);
    auto end = message.Finish();
    builder.Finish(end);
  };

  auto fn1 = [] {
    NaiveBuffer buf;
    buf << std::string("hello world");
    buf << 13;
  };

  {
    boost::timer timer;
    for (int i = 0; i < 1000; i++) {
      fn0();
    }
    LOG(INFO) << "timer: " << timer.elapsed();
  }

  {
    boost::timer timer;
    for (int i = 0; i < 1000; i++) {
      fn1();
    }
    LOG(INFO) << "timer: " << timer.elapsed();
  }

  {
    FlatBufferBuilder builder;
    auto hello = builder.CreateString("hello world");

    MessageRequestBuilder message(builder);
    message.add_greet(hello);
    message.add_v(13);
    auto end = message.Finish();
    builder.Finish(end);

    auto fn2 = [&] {
      auto message = GetRoot<MessageRequest>(builder.GetBufferPointer());
      return message->greet()->str() + std::to_string(message->v());
    };

    boost::timer timer;
    for (int i = 0; i < 1000; i++) {
      fn2();
    }
  }
}

}  // namespace tips