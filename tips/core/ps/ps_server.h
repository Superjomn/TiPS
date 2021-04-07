#pragma once

#include <absl/container/flat_hash_map.h>
#include <absl/container/inlined_vector.h>

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

Datatype ToDatatype(message::DataType dtype);

template <typename TABLE, typename PULL_ACCESS_METHOD, typename PUSH_ACCESS_METHOD>
class PsServer {
 public:
  using table_t              = TABLE;
  using key_t                = typename TABLE::key_t;
  using param_t              = typename TABLE::param_t;
  using value_t              = typename TABLE::value_t;
  using pull_access_method_t = PULL_ACCESS_METHOD;
  using push_access_method_t = PUSH_ACCESS_METHOD;
  using pull_access_agent_t  = PullAccessAgent<table_t, pull_access_method_t>;
  using push_access_agent_t  = PushAccessAgent<table_t, push_access_method_t>;

  using message_channel_t = Channel<ZmqMessage>;

  explicit PsServer(pull_access_method_t pull_access, push_access_method_t push_access, table_t* table)
      : pull_agent_(table, std::move(pull_access)),
        push_agent_(table, std::move(push_access)),
        table_(table),
        pull_channel_(std::make_shared<message_channel_t>()),
        push_channel_(std::make_shared<message_channel_t>()) {
    CHECK(table_);
    AddRpcService();
  }

  void Initialize() {
    if (!table_->Initialized()) table_->Initialize();
  }

  void Finalize() {
    if (!table_->Finalized()) table_->Finalize();
  }

 private:
  void AddRpcService();

  void InitPullService();
  void InitPushService();

  /**
   * Process a Pull request.
   */
  void PullTask(ZmqMessage&& zmq_msg);

  void PushTask(ZmqMessage&& zmq_msg);

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
    MPI_LOG << "Get a PullRequest";
    // CHECK(pull_channel_);
    // Just push the message to the channel, limit the workload of RPC threads.
    // pull_channel_->WriteMove(std::move(zmq_msg));

    PullTask(std::move(zmq_msg));
  };

  // Get a PullRequest message, and response the PullResponse.
  RpcCallback push_callback = [this](ZmqMessage&& zmq_msg) {
    CHECK(push_channel_);
    // Just push the message to the channel, limit the workload of RPC threads.
    push_channel_->WriteMove(std::move(zmq_msg));
  };

  RpcServer::Global().TryAddService(rpc::kPullService, pull_callback);
  RpcServer::Global().TryAddService(rpc::kPushService, push_callback);
}

template <typename TABLE, typename PULL_ACCESS_METHOD, typename PUSH_ACCESS_METHOD>
void PsServer<TABLE, PULL_ACCESS_METHOD, PUSH_ACCESS_METHOD>::InitPullService() {}

namespace {

// Holds the necessary datas while processing the task, we compose all the data in this structure for a single usage of
// share pointer. We create a PullTaskSnapshot for each PullTask.
template <typename key_t>
struct PullTaskSnapshot {
  ZmqMessage zmq_msg;

  const RpcMsgHead* msg_head() const { return GetMsgHead(zmq_msg); }
  const void* msg_buffer() const { return GetMsgContent(zmq_msg); }

  // shardid : the id of the local shard in the sparse table.
  // offset: the offset of this key in the original request order.
  absl::flat_hash_map<int /*shardid*/, std::vector<std::tuple<int /*offset*/, key_t, Datatype, int /*length*/>>>
      key_slots;

  std::vector<flatbuffers::Offset<message::KeyItem>> datas;

  // We construct the flatbuffers message directlly to avoid IO-copy from the raw data to message.
  flatbuffers::FlatBufferBuilder resp_builder;
  // message::PullResponseBuilder resp_msg{resp_builder};

  // To protect resp_builder and other shared data.
  std::mutex mu;

  // Should eval after all the key queries are finished.
  void TryDone(int finished) {
    finished_counter_ -= finished;

    if (finished_counter_ == 0) {
      auto* buffer = GetMsgContent(zmq_msg);
      CHECK(buffer);
      auto pull_request = flatbuffers::GetRoot<message::PullRequest>(buffer);

      auto keyitems = resp_builder.CreateVector(datas);

      // copy meta data from the request meta
      auto meta = message::CreateMessageMeta(
          resp_builder, pull_request->meta()->client_id(), pull_request->meta()->message_id());

      auto msg = message::CreatePullResponse(resp_builder, meta, keyitems);

      resp_builder.Finish(msg);

      // just send response back
      // TODO(Superjomn) optimize the io.

      LOG(INFO) << "Send back pull response";
      RpcServer::Global().SendResponse(*msg_head(), resp_builder.GetBufferPointer(), resp_builder.GetSize());
    }
  }

  PullTaskSnapshot(ZmqMessage&& zmq_msg, int key_count)
      : zmq_msg(std::move(zmq_msg)), datas(key_count), finished_counter_(key_count) {}

 private:
  std::atomic<int> finished_counter_;
};

}  // namespace

template <typename TABLE, typename PULL_ACCESS_METHOD, typename PUSH_ACCESS_METHOD>
void PsServer<TABLE, PULL_ACCESS_METHOD, PUSH_ACCESS_METHOD>::PullTask(ZmqMessage&& zmq_msg) {
  auto* msg_head = GetMsgHead(zmq_msg);
  auto* buffer   = GetMsgContent(zmq_msg);
  CHECK(buffer);
  auto pull_request = flatbuffers::GetRoot<message::PullRequest>(buffer);
  // NOTE Though the zmq_msg is moved latter, the memory address in message content is still valid.

  auto snapshot = std::make_shared<PullTaskSnapshot<key_t>>(std::move(zmq_msg), pull_request->keys()->size());

  int i = 0;

  for (key_t key : *pull_request->keys()) {
    int shard_id   = pull_agent_.ToShardId(key);
    Datatype dtype = ToDatatype(static_cast<message::DataType>(*pull_request->dtypes()[i].data()));
    int length     = *pull_request->lengths()[i].data();
    snapshot->key_slots[shard_id].emplace_back(std::make_tuple(i++, key, dtype, length));
  }

  for (auto& item : snapshot->key_slots) {
    // push tasks to shard channel
    auto& channel = table_->server_channel(item.first);
    auto& queries = item.second;

    channel.WriteMove([this, snapshot, queries] {
      value_t value;
      for (auto [offset, key, dtype, length] : queries) {
        pull_agent_.GetPullValue(key, value, dtype, length);

        {  // protect the resp_builder for it is shared by all the shard threads.
          std::lock_guard<std::mutex> lock(snapshot->mu);
          auto* data_addr = reinterpret_cast<uint8_t*>(value.template buffer());
          auto flat_data  = snapshot->resp_builder.CreateVector(data_addr, value.num_bytes());

          message::KeyItemBuilder key_builder(snapshot->resp_builder);
          key_builder.add_value(flat_data);
          key_builder.add_key(key);
          snapshot->datas[offset] = key_builder.Finish();
        }
      }

      snapshot->TryDone(queries.size());
    });
  }
}

namespace {

template <typename key_t>
struct PushTaskSnapshot {
  ZmqMessage zmq_msg;

  absl::flat_hash_map<key_t, std::vector<message::KeyItem*>> slots;

  const RpcMsgHead* msg_head() const { return GetMsgHead(zmq_msg); }
  const void* msg_buffer() const { return GetMsgContent(zmq_msg); }

  void TryDone(int finished) {}

  PushTaskSnapshot(ZmqMessage&& zmq_msg, int key_count) : zmq_msg(std::move(zmq_msg)), finished_counter_(key_count) {}

 private:
  std::atomic<int> finished_counter_;
};

}  // namespace

template <typename TABLE, typename PULL_ACCESS_METHOD, typename PUSH_ACCESS_METHOD>
void PsServer<TABLE, PULL_ACCESS_METHOD, PUSH_ACCESS_METHOD>::PushTask(ZmqMessage&& zmq_msg) {
  auto* msg_head = GetMsgHead(zmq_msg);
  auto* buffer   = GetMsgContent(zmq_msg);
  CHECK(buffer);
  auto push_request = flatbuffers::GetRoot<message::PushRequest>(buffer);
  // NOTE Though the zmq_msg is moved latter, the memory address in message content is still valid.

  auto snapshot = std::make_shared<PushTaskSnapshot<key_t>>(std::move(zmq_msg), push_request->data()->size());

  for (auto* item : *push_request->data()) {
    key_t key                 = item->key();
    int shard_id              = push_agent_->ToShardId(key);
    snapshot->slots[shard_id] = item;
  }

  for (auto& item : snapshot->slots) {
    int slot_id                                = item.first;
    const std::vector<message::KeyItem*>& rcds = item.second;

    auto& channel = table_->server_channel(slot_id);

    channel.WriteMove([this, snapshot, rcds] {
      for (auto& rcd : rcds) {
        auto* data = rcd->value();

        switch (rcd->dtype()) {
          case message::DataType_TF_FLOAT32:
            break;

          default:
            LOG(FATAL) << "Not supported type: " << rcd->dtype();
        }
      }
    });
  }
}

}  // namespace ps
}  // namespace tips
