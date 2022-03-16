#include "tips/core/ps/table_service.h"
#include "tips/core/ps/sparse_table.h"

namespace tips {

class SparseTableAccessAgent : public ps::TableAccessAgent {
 public:
  explicit SparseTableAccessAgent(ps::SparseTable* table) {}

  std::unique_ptr<FBS_TypeBufferOwned<ps::message::PullResponse>> Pull(const ps::message::PullRequest& msg) override {
    return std::unique_ptr<FBS_TypeBufferOwned<ps::message::PullResponse>>();
  }
  std::unique_ptr<FBS_TypeBufferOwned<ps::message::PushResponse>> Push(const ps::message::PushRequest& msg) override {
    return std::unique_ptr<FBS_TypeBufferOwned<ps::message::PushResponse>>();
  }
};

}  // namespace tips