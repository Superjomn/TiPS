#include "tips/core/common/flatbuffers_utils.h"

#include <gtest/gtest.h>

#include "tips/core/message/collective_messages_generated.h"

namespace tips {

TEST(FBS_TypeBufferOwned, move) {
  using RequestMessage = FBS_TypeBufferOwned<collective::message::RequestMessage>;

  RequestMessage message;
  ASSERT_FALSE(message.HasData());
}

}  // namespace tips