#include "tips/core/collective/coordinator.h"

namespace tips {
namespace collective {

bool IncreTensorCount(MessageTable& table, RequestMessage&& msg, int mpi_size) {
  auto name       = msg.msg().tensor_name()->str();
  auto table_iter = table.find(name);
  if (table_iter == table.end()) {
    table.emplace(name, std::vector<RequestMessage>({std::move(msg)}));
    table_iter = table.find(name);
  } else {
    table_iter->second.push_back(std::move(msg));
  }

  return table_iter->second.size() == mpi_size;
}

ResponseMessage ConstructResponseMessage(MessageTable& table, const std::string& name) {
  auto it = table.find(name);
  CHECK(it != table.end()) << "message table doesn't have item called " << name;

  const auto& requests = it->second;
  CHECK_GT(requests.size(), 0);

  bool error = false;
  std::stringstream error_stream;

  // Check that all data types are identical
  auto data_type = requests[0].msg().tensor_type();
  for (int i = 1; i < requests.size() && !error; i++) {
    auto type = requests[i].msg().tensor_type();
    if (data_type != type) {
      error = true;
      error_stream << "Mismatch data types found: " << data_type << " vs " << type << ".";
    }
  }

  // Check that all requested operations are the same.
  auto operation_type = requests[0].msg().request_type();
  for (int i = 1; i < requests.size() && !error; i++) {
    auto type = requests[0].msg().request_type();
    if (operation_type != type) {
      error = true;
      error_stream << "Mismatched operations found: " << operation_type << " vs " << type << ".";
    }
  }

  // If we are doing an allreduce, check that all the tensor shape are identical.
  tensorflow::TensorShape tensor_shape;
  if (operation_type == message::RequestType_ALLREDUCE) {
    for (int64_t d : *requests[0].msg().tensor_shape()) {
      tensor_shape.AddDim(d);
    }
    for (int i = 1; i < requests.size() && !error; i++) {
      tensorflow::TensorShape shape;
      for (int64_t d : *requests[i].msg().tensor_shape()) {
        shape.AddDim(d);
      }
      if (shape != tensor_shape) {
        error = true;
        error_stream << "Mismatched allreduce tensor shapes: " << tensor_shape.DebugString() << " vs "
                     << shape.DebugString();
      }
    }
  }

  // If we are doing an allgather, make sure all but the first dimension are the same. The first dimension may be
  // different and the output tensor is the sum of the first dimension.
  std::vector<int64_t> tensor_sizes(requests.size());
  if (operation_type == message::RequestType_ALLGATHER) {
    tensorflow::TensorShape tensor_shape;
    for (int64_t d : *requests[0].msg().tensor_shape()) {
      tensor_shape.AddDim(d);
    }

    if (tensor_shape.dims() == 0) {
      error = true;
      error_stream << "An empty tensor found";
    } else {
      tensor_sizes[requests[0].msg().request_rank()] = tensor_shape.dim_size(0);
    }

    for (int i = 1; i < requests.size() && !error; i++) {
      tensorflow::TensorShape request_shape;
      for (int64_t d : *requests[i].msg().tensor_shape()) {
        request_shape.AddDim(d);
      }
      if (tensor_shape.dims() != request_shape.dims()) {
        error = true;
        error_stream << "Mismatched allgather tensor shapes: rank " << tensor_shape.dims() << " vs "
                     << request_shape.dims();
      }

      for (int dim = 1; dim < tensor_shape.dims() && !error; dim++) {
        if (tensor_shape.dim_size(dim) != request_shape.dim_size(dim)) {
          error = true;
          error_stream << "Mismatched allgather tensor shapes: " << dim << "-th dimension "
                       << tensor_shape.dim_size(dim) << " vs " << request_shape.dim_size(dim);
        }
      }

      tensor_sizes[requests[i].msg().request_rank()] = request_shape.dim_size(0);
    }
  }

  // Clear all queued up requests for this tensor.
  table.erase(name);

  // construct response message
  {
    FlatBufferBuilder builder;
    auto tensor_name    = builder.CreateString(name);
    auto error_message  = builder.CreateString(error ? error_stream.str() : "");
    auto tensor_sizes_f = builder.CreateVector(tensor_sizes);

    message::ResponseMessageBuilder response(builder);
    response.add_tensor_name(tensor_name);

    if (error) {
      response.add_response_type(message::ResponseType_ERROR);
      response.add_error_message(error_message);
    } else if (operation_type == message::RequestType_ALLGATHER) {
      response.add_response_type(message::ResponseType_ALLGATHER);
      for (auto dim : tensor_sizes) {
        response.add_tensor_sizes(tensor_sizes_f);
      }
    } else if (operation_type == message::RequestType_ALLREDUCE) {
      response.add_response_type(message::ResponseType_ALLREDUCE);
    }

    builder.Finish(response.Finish());
    // TODO
    return ResponseMessage(builder.Release());
  }
}

RequestMessage CreateRequestMessage(int request_rank,
                                    message::RequestType request_type,
                                    message::DataType data_type,
                                    const std::string& tensor_name,
                                    const std::vector<int64_t>& tensor_shape) {
  FlatBufferBuilder builder;
  auto tensor_name_  = builder.CreateString(tensor_name);
  auto tensor_shape_ = builder.CreateVector(tensor_shape);

  auto message = message::RequestMessageBuilder(builder);
  message.add_request_rank(request_rank);
  message.add_request_type(request_type);
  message.add_tensor_type(data_type);
  message.add_tensor_name(tensor_name_);
  message.add_tensor_shape(tensor_shape_);

  builder.Finish(message.Finish());

  return RequestMessage(builder.Release());
}

void EnqueueTensorCollective(const OpRecord& record, message::RequestType request_type) {
  const Tensor* in_tensor = record.in_tensor;
  std::vector<int64_t> shape;
  for (int i = 0; i < in_tensor->shape().dims(); i++) {
    shape.push_back(in_tensor->shape().dim_size(i));
  }

  auto message = CreateRequestMessage(record.rank, request_type, record.dtype, record.name, shape);

  std::lock_guard<std::mutex> lock(CollectiveState::Global().mu);
  CollectiveState::Global().tensor_table.emplace(record.name, record);
  CollectiveState::Global().message_queue.push(std::move(message));
}

}  // namespace collective
}  // namespace tips