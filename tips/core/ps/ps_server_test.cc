#include "tips/core/ps/ps_server.h"
#include <vector>
#include "tips/core/common/any_vec.h"
#include "tips/core/common/vec.h"
#include "tips/core/mpi/tips_mpi.h"
#include "tips/core/operations.h"
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

void TestBasic() {
  struct Value {
    float x;

    void* data() { return &x; }
    size_t num_bytes() const { return sizeof(float); }

    Value operator*(float lr) const {
      Value res;
      res.x *= lr;
      return res;
    }

    Value operator+(const Value& other) const {
      Value res;
      res.x += other.x;
      return res;
    }

    Value& operator+=(const Value& other) {
      x += other.x;
      return *this;
    }

    std::ostream& operator<<(std::ostream& os) const {
      os << x;
      return os;
    }
  };

  using val_t         = AnyVec;
  using key_t         = uint64_t;
  using param_t       = AnyVec;
  using table_t       = SparseTable<key_t, val_t>;
  using pull_access_t = SparseTablePullAccess<key_t, param_t, val_t>;
  using push_access_t = SparseTableSgdPushAccess<key_t, param_t, param_t>;
  using server_t      = PsServer<table_t, pull_access_t, push_access_t>;

  table_t table;
  pull_access_t pull_access(&table);
  push_access_t push_access(&table, 1);

  server_t server(pull_access, push_access, &table);
  server.Initialize();

  auto pull_request = CreatePullRequest({1, 2, 3});

  auto* service = RpcServer::Global().LookupService(rpc::kPullService);

  std::condition_variable cv;
  std::mutex mu;

  RpcServer::Global().SendRequest(0, service, pull_request.data(), pull_request.size(), [&](ZmqMessage&& zmq_msg) {
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
        CHECK_NEAR(val[i], 0, 1e-5);
      }
    }

    cv.notify_one();
  });

  std::unique_lock<std::mutex> lock(mu);
  cv.wait(lock);

  server.Finalize();

  LOG(INFO) << "end test";
}

}  // namespace ps
}  // namespace tips

int main() {
  tips_init();

  ps::TestBasic();

  tips_shutdown();
}