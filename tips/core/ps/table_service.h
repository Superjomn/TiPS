#pragma once
#include <memory>
#include "tips/core/common/common.h"
#include "tips/core/common/flatbuffers_utils.h"
#include "tips/core/message/ps_messages_generated.h"

namespace tips {
namespace ps {

enum class PsTableType {
  kSparse     = 0,
  kDense_FP32 = 1,
  kDense_FP64 = 2,
};

enum class PsOptimizeStrategyType {
  kSGD = 0,
};

/**
 * TableService: A service for listening for PULL PUSH requests.
 */
class TableService {
 public:
  //! Create a new table.
  void AddTable(uint64_t talbe_id, PsTableType table_type, int num_shards);
};

}  // namespace ps
}  // namespace tips
