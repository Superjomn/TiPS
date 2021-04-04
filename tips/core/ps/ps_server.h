#pragma once

#include "tips/core/common/channel.h"
#include "tips/core/common/common.h"
#include "tips/core/common/flatbuffers_utils.h"
#include "tips/core/common/naive_rpc.h"
#include "tips/core/message/ps_messages_generated.h"

namespace tips {
namespace ps {

using PushRequest  = FBS_TypeBufferOwned<message::PushRequest>;
using PullRequest  = FBS_TypeBufferOwned<message::PullRequest>;
using PushResponse = FBS_TypeBufferOwned<message::PushResponse>;
using PullResponse = FBS_TypeBufferOwned<message::PullResponse>;

template <typename TABLE, typename ACCESS_METHOD>
class PsServer {
 public:
  using table_t         = TABLE;
  using key_t           = typename TABLE::value_t;
  using value_t         = typename TABLE::value_t;
  using access_method_t = ACCESS_METHOD;

  explicit PsServer(access_method_t method) : access_method_(method) {}

  void Initialize();

  void Finalize();

 private:
  void AddRpcService();

  table_t table_;

  Channel<PullRequest> pull_channel_;
  Channel<PushRequest> push_channel_;

  access_method_t access_method_;
};

template <typename TABLE, typename ACCESS_METHOD>
void PsServer<TABLE, ACCESS_METHOD>::AddRpcService() {
  // Get a PullRequest message, and response the PullResponse.
  RpcCallback pull_callback = [this](const RpcMsgHead& head, uint8_t* data) {
    auto pull_message = flatbuffers::GetRoot<message::PullRequest>(data);
    // TODO(Superjomn) Zero-copy here.
  };
}

}  // namespace ps
}  // namespace tips
