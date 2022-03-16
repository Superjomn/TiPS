#pragma once
#include <memory>
#include "tips/core/common/common.h"
#include "tips/core/common/flatbuffers_utils.h"
#include "tips/core/message/ps_messages_generated.h"

namespace tips {
namespace ps {

/**
 * Interface for the access method for any dense tables.
 */
class TableAccessAgent {
 public:
  virtual std::unique_ptr<FBS_TypeBufferOwned<ps::message::PullResponse>> Pull(const ps::message::PullRequest& msg) = 0;

  virtual std::unique_ptr<FBS_TypeBufferOwned<ps::message::PushResponse>> Push(const ps::message::PushRequest& msg) = 0;
};

/**
 * Create a TableAccessAgent for a specific type of Table.
 */
template <typename TableType>
std::unique_ptr<TableAccessAgent> CreateTableAccessAgent(TableType* table);

}  // namespace ps
}  // namespace tips
