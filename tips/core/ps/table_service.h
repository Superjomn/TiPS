#pragma once
#include <memory>
#include "tips/core/common/common.h"
#include "tips/core/common/flatbuffers_utils.h"
#include "tips/core/message/ps_messages_generated.h"

namespace tips {

enum class PsTableType {
  kSparse     = 0,
  kDense_FP32 = 1,
  kDense_FP64 = 2,
};

enum class PsOptimizeStrategyType {
  kSGD = 0,
};

/**
 * Interface for the access method for any dense tables.
 */
class TableAccessAgent {
 public:
  virtual std::unique_ptr<FBS_TypeBufferOwned<ps::message::PullResponse>> Pull(const ps::message::PullRequest& msg) = 0;

  virtual std::unique_ptr<FBS_TypeBufferOwned<ps::message::PushResponse>> Push(const ps::message::PushRequest& msg) = 0;
};

/**
 * TableService: A service for listening for PULL PUSH requests.
 */
class TableService {
 public:
  //! Create a new table.
  void AddTable(uint64_t talbe_id, PsTableType table_type, int num_shards);
};

}  // namespace tips
