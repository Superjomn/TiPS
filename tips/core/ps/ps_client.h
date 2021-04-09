#pragma once

#include "tips/core/common/any_vec.h"
#include "tips/core/common/common.h"
#include "tips/core/common/naive_rpc.h"
#include "tips/core/message/ps_messages_generated.h"
#include "tips/core/ps/ps_utils.h"
#include "tips/core/ps/route.h"
#include "tips/core/rpc_service_names.h"

namespace tips {
namespace ps {

//! A easy-to-use-in-C++ struct of pull request.
// The flatbuffers is too trivial to use.
struct PullRequest {
  struct Record {
    Record(uint64_t key, Datatype dtype, int length) : key(key), dtype(dtype), length(length) {}

    uint64_t key;
    Datatype dtype;
    int length{};
  };
  std::vector<Record> data;
};

struct PushRequest {
  struct Record {
    uint64_t key;
    AnyVec vec;
  };
  std::vector<Record> data;
};

struct PullCache {
  ZmqMessage msg;  // Hold the original message to avoid copy Vec.

  using Record = AnyVec;
  // NOTE TODO(Superjomn) The order remains the same with the original keys order in the pull request, so we do not need
  // a map here if the offset is recorded.
  std::unordered_map<uint64_t, Record> data;
};

template <typename TABLE>
class PsClient {
 public:
  using table_t    = TABLE;
  using key_t      = typename table_t::key_t;
  using param_t    = typename table_t::param_t;
  using callback_t = std::function<void()>;

  PsClient(const Route& route, table_t* table) : route_(route), table_(table) {
    CHECK(table_);
    pull_service_ = RpcServer::Global().LookupService(rpc::kPullService);
    push_service_ = RpcServer::Global().LookupService(rpc::kPushService);
    CHECK(pull_service_);
    CHECK(push_service_);
  }

  bool Pull(const PullRequest& req, PullCache* cache, callback_t done);

  bool Push(const PushRequest& req, callback_t done);

 private:
  flatbuffers::DetachedBuffer MakePullRequestBuffer(absl::Span<PullRequest::Record> req) const;
  flatbuffers::DetachedBuffer MakePushRequestBuffer(absl::Span<PushRequest::Record> req) const;

 private:
  table_t* table_{};

  // A cache point to the global RPC server's services.
  RpcService* pull_service_{};
  RpcService* push_service_{};

  const Route& route_;
};

template <typename TABLE>
bool PsClient<TABLE>::Pull(const PullRequest& req, PullCache* cache, callback_t done) {
  std::unordered_map<int, PullRequest> slots;  // shardid to request
  for (auto& rcd : req.data) {
    int serverid    = table_->template ToServerId(rcd.key);  // group rank
    auto& mpi_group = route_.GetGroup<Route::NodeKind::PS_SERVER>();
    int global_rank = mpi_group.ToWorldRank(serverid);
    slots[global_rank].data.push_back(rcd);
  }

  // send to servers.
  for (auto& item : slots) {
    auto buffer = MakePullRequestBuffer(absl::Span<PullRequest::Record>(item.second.data));
    MPI_LOG << "Send PULL request to PsServer #" << item.first;
    RpcServer::Global().SendRequest(
        item.first, pull_service_, buffer.data(), buffer.size(), [done, cache](ZmqMessage&& zmq_msg) {
          LOG(INFO) << "Pull request done";
          cache->msg = std::move(zmq_msg);

          // load the parameters
          auto pull_response = flatbuffers::GetRoot<message::PullResponse>(GetMsgContent(cache->msg));
          for (auto* item : *pull_response->data()) {
            Datatype dtype = ToDatatype(item->dtype());
            AnyVec vec(
                dtype, item->value()->size() / DatatypeNumBytes(dtype), const_cast<uint8_t*>(item->value()->data()));
            cache->data[item->key()] = std::move(vec);
          }

          done();
        });
  }
}

template <typename TABLE>
bool PsClient<TABLE>::Push(const PushRequest& req, callback_t done) {
  std::unordered_map<int, std::vector<PushRequest::Record>> slots;  // shardid to request
  for (auto& rcd : req.data) {
    int shardid     = table_->template ToShardId(rcd.key);  // group rank
    auto& mpi_group = route_.GetGroup<Route::NodeKind::PS_SERVER>();
    int global_rank = mpi_group.ToWorldRank(shardid);
    slots[global_rank].push_back(rcd);
  }

  // send to servers.
  for (auto& item : slots) {
    auto buffer = MakePushRequestBuffer(absl::Span<PushRequest::Record>(item.second));
    RpcServer::Global().SendRequest(
        item.first, push_service_, buffer.GetBuffer(), buffer.GetSize(), [done](ZmqMessage&&) {
          LOG(INFO) << "Push request done";
          done();
        });
  }
}

template <typename TABLE>
flatbuffers::DetachedBuffer PsClient<TABLE>::MakePullRequestBuffer(absl::Span<PullRequest::Record> req) const {
  flatbuffers::FlatBufferBuilder builder;

  auto meta = message::CreateMessageMeta(builder, mpi_rank(), 0 /*messageid*/);
  std::vector<key_t> keys;
  std::vector<uint32_t> lens;
  std::vector<short> dtypes;
  for (auto& d : req) {
    keys.push_back(d.key);
    lens.push_back(d.length);
    dtypes.push_back(ToMessageDataType(d.dtype));
  }

  auto _keys   = builder.CreateVector(keys);
  auto _dtypes = builder.CreateVector(dtypes);
  auto _lens   = builder.CreateVector(lens);

  auto msg = message::CreatePullRequest(builder, meta, _keys, _dtypes, _lens);
  builder.Finish(msg);

  return builder.Release();
}

template <typename TABLE>
flatbuffers::DetachedBuffer PsClient<TABLE>::MakePushRequestBuffer(absl::Span<PushRequest::Record> req) const {
  flatbuffers::FlatBufferBuilder builder;
  std::vector<flatbuffers::Offset<message::KeyItem>> datas;
  datas.reserve(req.size());

  for (auto& rcd : req) {
    auto v    = builder.CreateVector(reinterpret_cast<uint8_t*>(rcd.vec.buffer()), rcd.vec.num_bytes());
    auto item = message::CreateKeyItem(builder, rcd.key, ToMessageDataType(rcd.vec.dtype()), v);
    datas.push_back(item);
  }

  auto meta = message::CreateMessageMeta(builder, mpi_rank(), 0 /*messageid*/);

  auto _datas = builder.CreateVector(datas);
  auto msg    = message::CreatePushRequest(builder, meta, _datas);

  builder.Finish(msg);

  return builder.Release();
}

}  // namespace ps
}  // namespace tips
