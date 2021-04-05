#pragma once

#include "tips/core/common/channel.h"
#include "tips/core/common/common.h"
#include "tips/core/common/flatbuffers_utils.h"
#include "tips/core/common/naive_rpc.h"
#include "tips/core/message/ps_messages_generated.h"
#include "tips/core/ps/access_method.h"
#include "tips/core/ps/table.h"
#include "tips/core/rpc_service_names.h"

namespace tips {
namespace ps {

using PushRequest  = FBS_TypeBufferOwned<message::PushRequest>;
using PullRequest  = FBS_TypeBufferOwned<message::PullRequest>;
using PushResponse = FBS_TypeBufferOwned<message::PushResponse>;
using PullResponse = FBS_TypeBufferOwned<message::PullResponse>;

template <typename TABLE, typename PULL_ACCESS_METHOD, typename PUSH_ACCESS_METHOD>
class PsServer {
 public:
  using table_t              = TABLE;
  using key_t                = typename TABLE::value_t;
  using param_t              = typename TABLE::param_t;
  using value_t              = typename TABLE::value_t;
  using pull_access_method_t = PULL_ACCESS_METHOD;
  using push_access_method_t = PUSH_ACCESS_METHOD;
  using pull_access_agent_t  = PullAccessAgent<table_t, pull_access_method_t>;
  using push_access_agent_t  = PushAccessAgent<table_t, push_access_method_t>;

  using message_channel_t = Channel<ZmqMessage>;

  explicit PsServer(pull_access_method_t pull_access, push_access_method_t push_access, Table* table)
      : pull_agent_(table, std::move(pull_access)), push_agent_(table, std::move(push_access)), table_(table) {
    CHECK(table_);
  }

  void Initialize();

  void Finalize();

 private:
  void AddRpcService();

  void InitPullService();
  void InitPushService();

  void PullTask(const ZmqMessage& zmq_msg) {
    auto* msg_head = GetMsgHead(zmq_msg);
    auto* buffer   = GetMsgContent(zmq_msg);
    CHECK(buffer);
    auto pull_request = flatbuffers::GetRoot<message::PullRequest>(buffer);

    // Lookup the value and construct the flatbuffers message.
    flatbuffers::FlatBufferBuilder resp_builder;
    message::PullResponseBuilder resp_msg(resp_builder);

    std::vector<flatbuffers::Offset<message::KeyItem>> datas(pull_request->keys()->size());

    // copy meta data from the request meta
    message::MessageMetaBuilder meta_builder(resp_builder);
    meta_builder.add_client_id(pull_request->meta()->client_id());
    meta_builder.add_message_id(pull_request->meta()->message_id());
    resp_msg.add_meta(meta_builder.Finish());

    int i = 0;
    for (key_t key : *pull_request->keys()) {
      message::KeyItemBuilder key_builder(resp_builder);

      value_t value;
      pull_agent_.GetPullValue(key, value);

      key_builder.add_key(key);
      key_builder.add_value(resp_builder.CreateVector<char>(value.data(), value.num_bytes()));

      datas[i++] = key_builder.Finish();
    }

    auto keyitems = resp_builder.CreateVector(datas);
    resp_msg.add_data(keyitems);
    resp_builder.Finish(resp_msg.Finish());

    // just send response back
    // TODO(Superjomn) optimize it.

    LOG(INFO) << "Send back pull response";
    RpcServer::Global().SendResponse(*msg_head, resp_builder.GetBufferPointer(), resp_builder.GetSize());
  }

  std::shared_ptr<message_channel_t> pull_channel_;
  std::shared_ptr<message_channel_t> push_channel_;

  pull_access_agent_t pull_agent_;
  push_access_agent_t push_agent_;

  Table* table_{};
};

template <typename TABLE, typename PULL_ACCESS_METHOD, typename PUSH_ACCESS_METHOD>
void PsServer<TABLE, PULL_ACCESS_METHOD, PUSH_ACCESS_METHOD>::AddRpcService() {
  // Get a PullRequest message, and response the PullResponse.
  RpcCallback pull_callback = [this](ZmqMessage&& zmq_msg) {
    CHECK(pull_channel_);
    // Just push the message to the channel, limit the workload of RPC threads.
    pull_channel_->WriteMove(std::move(zmq_msg));
  };

  // Get a PullRequest message, and response the PullResponse.
  RpcCallback push_callback = [this](ZmqMessage&& zmq_msg) {
    CHECK(push_channel_);
    // Just push the message to the channel, limit the workload of RPC threads.
    push_channel_->WriteMove(std::move(zmq_msg));
  };

  RpcServer::Global().AddService(rpc::kPullService, pull_callback);
  RpcServer::Global().AddService(rpc::kPushService, push_callback);
}

template <typename TABLE, typename PULL_ACCESS_METHOD, typename PUSH_ACCESS_METHOD>
void PsServer<TABLE, PULL_ACCESS_METHOD, PUSH_ACCESS_METHOD>::InitPullService() {}

}  // namespace ps
}  // namespace tips
