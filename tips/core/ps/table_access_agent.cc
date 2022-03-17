#include "tips/core/ps/table_access_agent.h"
#include "tips/core/ps/sparse_table.h"

namespace tips {
namespace ps {

class SparseTableAccessAgent : public TableAccessAgent {
 public:
  explicit SparseTableAccessAgent(SparseTable* table) : table_(table) {}

  std::unique_ptr<FBS_TypeBufferOwned<ps::message::PullResponse>> Pull(const message::PullRequest& msg) override {
    return std::unique_ptr<FBS_TypeBufferOwned<ps::message::PullResponse>>();
  }

  std::unique_ptr<FBS_TypeBufferOwned<ps::message::PushResponse>> Push(const message::PushRequest& msg) override {
    return std::unique_ptr<FBS_TypeBufferOwned<ps::message::PushResponse>>();
  }

 private:
  SparseTable* table_;
};

}  // namespace ps
}  // namespace tips