#include "tips/core/ps/ps_server.h"

namespace tips {
namespace ps {

Datatype ToDatatype(message::DataType dtype) {
  switch (dtype) {
    case message::DataType_TF_INT32:
      return Datatype::kInt32;
    case message::DataType_TF_INT64:
      return Datatype::kInt64;
    case message::DataType_TF_FLOAT32:
      return Datatype::kFp32;
    case message::DataType_TF_FLOAT64:
      return Datatype::kFp64;
    default:
      LOG(FATAL) << "Not supported type";

      return Datatype::kFp32;
  }
}

}  // namespace ps
}  // namespace tips
