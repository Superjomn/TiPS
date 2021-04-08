#include "tips/core/ps/ps_utils.h"
#include "tips/core/ps/ps_server.h"

namespace tips {
namespace ps {

Datatype ToDatatype(message::DataType dtype) {
  switch (dtype) {
    case tips::ps::message::DataType_TF_INT32:
      return tips::Datatype::kInt32;
    case tips::ps::message::DataType_TF_INT64:
      return tips::Datatype::kInt64;
    case tips::ps::message::DataType_TF_FLOAT32:
      return tips::Datatype::kFp32;
    case tips::ps::message::DataType_TF_FLOAT64:
      return tips::Datatype::kFp64;
    default:
      LOG(FATAL) << "Not supported type";
  }
  return Datatype::kFp32;
}

message::DataType ToMessageDataType(tips::Datatype dtype) {
  switch (dtype) {
    case tips::Datatype::kFp32:
      return tips::ps::message::DataType_TF_FLOAT32;
    case tips::Datatype::kFp64:
      return tips::ps::message::DataType_TF_FLOAT64;
    case tips::Datatype::kInt32:
      return tips::ps::message::DataType_TF_INT32;
    case tips::Datatype::kInt64:
      return tips::ps::message::DataType_TF_INT64;
    default:
      LOG(FATAL) << "Not supported type";
  }
  return message::DataType_TF_FLOAT32;
}

}  // namespace ps
}  // namespace tips