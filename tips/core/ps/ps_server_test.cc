#include "tips/core/ps/ps_server.h"

#include <vector>

#include "ps_utils.h"
#include "tips/core/common/buffer.h"
#include "tips/core/common/vec.h"
#include "tips/core/mpi/tips_mpi.h"
#include "tips/core/operations.h"
#include "tips/core/ps/route.h"
#include "tips/core/ps/sparse_access_method.h"
#include "tips/core/ps/sparse_table.h"
#include "tips/core/rpc_service_names.h"

using namespace tips;

namespace tips {
namespace ps {

flatbuffers::DetachedBuffer CreatePullRequest(const std::vector<uint64_t>& keys) {
  flatbuffers::FlatBufferBuilder builder;

  auto meta    = message::CreateMessageMeta(builder, mpi_rank(), 0);
  auto _keys   = builder.CreateVector(keys);
  auto dtypes  = builder.CreateVector(std::vector<int16_t>(keys.size(), message::DataType_TF_FLOAT32));
  auto lengths = builder.CreateVector(std::vector<uint32_t>(keys.size(), 10));

  auto pull_request = message::CreatePullRequest(builder, meta, _keys, dtypes, lengths);
  builder.Finish(pull_request);

  return builder.Release();
}

flatbuffers::DetachedBuffer CreatePushRequest(const std::vector<uint64_t>& keys, const std::vector<Buffer>& vecs) {
  flatbuffers::FlatBufferBuilder builder;

  std::vector<flatbuffers::Offset<message::KeyItem>> datas;
  for (int i = 0; i < keys.size(); i++) {
    auto key  = keys[i];
    auto& vec = vecs[i];

    auto v   = builder.CreateVector(reinterpret_cast<const uint8_t*>(vec.buffer()), vec.num_bytes());
    auto rcd = message::CreateKeyItem(builder, key, ToMessageDataType(vec.dtype()), v);
    datas.push_back(rcd);
  }

  auto meta   = message::CreateMessageMeta(builder, mpi_rank(), 0);
  auto _datas = builder.CreateVector(datas);
  auto msg    = message::CreatePushRequest(builder, meta, _datas);
  builder.Finish(msg);

  return builder.Release();
}

void TestBasic() {
  using val_t         = Buffer;
  using key_t         = uint64_t;
  using param_t       = Buffer;
  using table_t       = SparseTable<key_t, val_t>;
  using pull_access_t = SparseTablePullAccess<key_t, param_t, val_t>;
  using push_access_t = SparseTableSgdPushAccess<key_t, param_t, param_t>;
  using server_t      = PsServer<table_t, pull_access_t, push_access_t>;

  Route::Global().RegisterNode<Route::NodeKind::PS_SERVER>(0);
  Route::Global().Initialize();

  table_t table(Route::Global().GetGroup<Route::NodeKind::PS_SERVER>());
  pull_access_t pull_access(&table);
  push_access_t push_access(&table, 1);

  server_t server(pull_access, push_access, &table);
  server.StartService();

  auto pull_request = CreatePullRequest({1, 2, 3});

  auto* pull_service = RpcServer::Global().LookupService(rpc::kPullService);
  auto* push_service = RpcServer::Global().LookupService(rpc::kPushService);

  std::condition_variable cv;
  std::mutex mu;

  RpcServer::Global().SendRequest(0, pull_service, pull_request.data(), pull_request.size(), [&](ZmqMessage&& zmq_msg) {
    LOG(INFO) << "Get Pull response";

    auto* head   = GetMsgHead(zmq_msg);
    auto* buffer = GetMsgContent(zmq_msg);
    CHECK(buffer);

    CHECK_EQ(head->client_id, mpi_rank());

    auto pull_response = flatbuffers::GetRoot<message::PullResponse>(buffer);
    std::vector<std::pair<key_t, float>> datas(pull_response->data()->size() / sizeof(float));
    for (const auto& item : *pull_response->data()) {
      LOG(INFO) << "responsed key: " << item->key();
      auto* val = reinterpret_cast<const float*>(item->value()->data());
      LOG(INFO) << "size: " << item->value()->size();
      for (int i = 0; i < 10; i++) {
        CHECK_NEAR(val[i], 0.f, 1e-5);
      }
    }

    cv.notify_one();
  });

  {
    std::unique_lock<std::mutex> lock(mu);
    cv.wait(lock);
  }

  std::vector<Buffer> vecs;
  vecs.emplace_back(DatatypeTypetrait<float>(), 10);
  vecs.emplace_back(DatatypeTypetrait<float>(), 10);
  for (int i = 0; i < 10; i++) vecs[0].mutable_data<float>()[i] = 1.f;
  for (int i = 0; i < 10; i++) vecs[1].mutable_data<float>()[i] = 2.f;

  auto push_message = CreatePushRequest({1, 2}, vecs);
  RpcServer::Global().SendRequest(0, push_service, push_message.data(), push_message.size(), [&](ZmqMessage&&) {
    LOG(INFO) << "get push response";
    cv.notify_one();
  });
  {
    std::unique_lock<std::mutex> lock(mu);
    cv.wait(lock);
  }

  server.StopService();

  Route::Global().Finalize();

  LOG(INFO) << "end test";
}

}  // namespace ps
}  // namespace tips

int main() {
  tips_init();

  ps::TestBasic();

  tips_shutdown();
}