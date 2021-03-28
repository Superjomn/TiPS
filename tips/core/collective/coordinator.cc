#include "tips/core/collective/coordinator.h"

#include <absl/types/span.h>

#include "tips/core/utils/string.h"

namespace tips {
namespace collective {
using tensorflow::TensorShape;
using namespace std::chrono_literals;

const char* kShutdownRpcServiceName    = "collective_shutdown";
const char* kCoordinatorRpcServiceName = "collective_coordinator";

bool IncreTensorCount(MessageTable& message_table, RequestMessage&& msg, int mpi_size) {
  CHECK(msg.HasData()) << "Message is invalid";
  auto name = msg.msg().tensor_name()->str();
  MPI_LOG << "IncreTensorCount: " << msg.msg().tensor_name()->str();
  auto table_iter = message_table.find(name);
  if (table_iter == message_table.end()) {
    message_table[name].emplace_back(std::move(msg));
    table_iter = message_table.find(name);
  } else {
    if (!table_iter->second.empty()) {
      CHECK_EQ(msg.msg().request_type(), table_iter->second.back().msg().request_type())
          << "Request type should match in a collective operation, " << msg.msg().request_type() << " vs "
          << table_iter->second.back().msg().request_type();
    }
    table_iter->second.emplace_back(std::move(msg));
  }

  MPI_LOG << "tensor record " << table_iter->first << " count: " << table_iter->second.size();
  for (int i = 0; i < table_iter->second.size(); i++) {
    CHECK(table_iter->second[i].HasData()) << i << "-th Message is invalid";
  }

  return table_iter->second.size() == mpi_size;
}

std::vector<int64_t> GatherFirstRankSizes(absl::Span<RequestMessage> requests,
                                          bool& error,
                                          std::stringstream& error_stream) {
  CHECK_EQ(requests.size(), mpi_size());
  std::vector<int64_t> tensor_sizes(requests.size(), 0);

  MPI_LOG << "0-th request shape: "
          << ToString(absl::Span<const int64_t>(requests[0].msg().tensor_shape()->data(),
                                                size_t(requests[0].msg().tensor_shape()->size())));
  MPI_LOG << "0-th request-rank: " << requests[0].msg().request_rank();

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
        error_stream << "Mismatched allgather tensor shapes: " << dim << "-th dimension " << tensor_shape.dim_size(dim)
                     << " vs " << request_shape.dim_size(dim);
      }
    }

    tensor_sizes[requests[i].msg().request_rank()] = request_shape.dim_size(0);
  }

  CHECK_EQ(tensor_sizes.size(), mpi_size());
  MPI_LOG << "GatherFirstRankSizes: " << ToString(absl::Span<int64_t>(tensor_sizes.data(), tensor_sizes.size()));
  return tensor_sizes;
}

ResponseMessage ConstructResponseMessage(MessageTable& table, const std::string& name) {
  MPI_LOG << "constructing the response message";
  auto it = table.find(name);
  CHECK(it != table.end()) << "message table doesn't have item called " << name;

  const auto& requests = it->second;
  CHECK_GT(requests.size(), 0);
  MPI_LOG << "requests.size: " << requests.size();
  MPI_LOG << "to visit request";
  CHECK(requests[0].HasData()) << "request message is invalid";
  MPI_LOG << requests[0].msg().tensor_name()->str();

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

  if (error) {
    MPI_LOG << "get error in construct response";
  }

  // If we are doing an allreduce, check that all the tensor shape are identical.
  tensorflow::TensorShape tensor_shape;
  if (operation_type == message::RequestType_ALLREDUCE || operation_type == message::RequestType_BROADCAST) {
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
        error_stream << "Mismatched " << (operation_type == message ::RequestType_BROADCAST ? "broadcast" : "allreduce")
                     << " tensor shapes: " << tensor_shape.DebugString() << " vs " << shape.DebugString();
      }
    }
  }

  // If we are doing an allgather, make sure all but the first dimension are the same. The first dimension may be
  // different and the output tensor is the sum of the first dimension.
  MPI_LOG << "requests.size: " << requests.size();
  std::vector<int64_t> tensor_rank0_sizes(requests.size());
  if (operation_type == message::RequestType_ALLGATHER) {
    tensor_rank0_sizes =
        GatherFirstRankSizes(absl::Span<RequestMessage>(it->second.data(), it->second.size()), error, error_stream);
  }

  // construct response message
  {
    FlatBufferBuilder builder;
    auto tensor_name    = builder.CreateString(name);
    auto error_message  = builder.CreateString(error ? error_stream.str() : "");
    auto tensor_sizes_f = builder.CreateVector(tensor_rank0_sizes);

    message::ResponseMessageBuilder response(builder);
    response.add_tensor_name(tensor_name);

    if (error) {
      response.add_response_type(message::ResponseType_ERROR);
      response.add_error_message(error_message);
    } else if (operation_type == message::RequestType_ALLGATHER) {
      response.add_response_type(message::ResponseType_ALLGATHER);
      response.add_tensor_sizes(tensor_sizes_f);
    } else if (operation_type == message::RequestType_ALLREDUCE) {
      CHECK(!tensor_rank0_sizes.empty()) << "Tensor size is empty";
      response.add_response_type(message::ResponseType_ALLREDUCE);
    } else if (operation_type == message::RequestType_BROADCAST) {
      response.add_response_type(message::ResponseType_BROADCAST);
    } else {
      LOG(FATAL) << "Not supported request type: " << operation_type;
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
  CHECK(!tensor_shape.empty());
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
  MPI_LOG << "message.name: " << message.msg().tensor_name()->str();

  MPI_LOG << " enqueue record [" << record.name << "] to the queue";
  CHECK(record.callback);

  {
    std::lock_guard<std::mutex> lock(CollectiveState::Global().mu);
    CollectiveState::Global().tensor_table.emplace(record.name, record);
  }

  // TODO(Superjomn) avoid creating RpcMsgHead here.
  RpcMsgHead head;
  CollectiveState::Global().message_queue->WriteMove(std::make_pair(head, std::move(message)));
}

void PerformCollectiveOp(OpRecord* op_record,
                         message::ResponseType response_type,
                         const std::string name,
                         const std::string& error_msg) {
  MPI_LOG << "Perform collective!";
  CHECK(response_type == message::ResponseType_ALLREDUCE || response_type == message::ResponseType_ALLGATHER ||
        response_type == message::ResponseType_BROADCAST || response_type == message::ResponseType_ERROR);

  Status status;
  auto dtype = op_record->dtype;

  if (response_type == message::ResponseType_ALLREDUCE) {
    CHECK(op_record->out_tensor);
    MPI_LOG << "Run MPI ALLREDUCE ...";
    switch (dtype) {
      case message::DataType_TF_INT32:
        MPI_LOG << "Allreduce int32 ...";
        status = AllreduceCpu<int32_t>(op_record->in_tensor, op_record->out_tensor, CollectiveOpKind::SUM);
        break;
      case message::DataType_TF_FLOAT32:
        MPI_LOG << "Allreduce float32 ...";
        status = AllreduceCpu<float>(op_record->in_tensor, op_record->out_tensor, CollectiveOpKind::SUM);
        break;
    }
  } else if (response_type == message::ResponseType_BROADCAST) {
    CHECK(op_record->out_tensor);
    switch (dtype) {
      case message::DataType_TF_INT32:
        // NOTE root_rank should be passed in.
        status = BroadcastCpu<int32_t>(op_record->out_tensor, 0);
        break;
      case message::DataType_TF_FLOAT32:
        status = BroadcastCpu<float>(op_record->out_tensor, 0);
        break;
    }
  } else if (response_type == message::ResponseType_ALLGATHER) {
    MPI_LOG << "Run MPI ALLGATHER ...";

    std::vector<int32_t> first_rank_sizes(op_record->sizes_vec.size());
    CHECK(!op_record->sizes_vec.empty());
    MPI_LOG << "record.sizes_vec: "
            << ToString<int64_t>(absl::Span<int64_t>(op_record->sizes_vec.data(), op_record->sizes_vec.size()));
    for (int i = 0; i < op_record->sizes_vec.size(); i++) first_rank_sizes[i] = op_record->sizes_vec[i];

    // Allocate output tensor for that the Allgather output's shape is known now.
    TensorShape shape = op_record->in_tensor->shape();
    shape.RemoveDim(0);
    shape.InsertDim(
        0, std::accumulate(op_record->sizes_vec.begin(), op_record->sizes_vec.end(), 0, [](int64_t a, int64_t b) {
          return a + b;
        }));
    MPI_LOG << "allgather output shape: " << shape.DebugString();
    CHECK(op_record->op_context);
    CHECK(!op_record->out_tensor);
    auto status = op_record->op_context->allocate_output(0, shape, &op_record->out_tensor);
    OP_REQUIRES_OK_ASYNC(op_record->op_context, status, [&] { op_record->callback(status); });

    CHECK(op_record->out_tensor);
    CHECK(op_record->in_tensor);
    switch (dtype) {
      case message::DataType_TF_INT32:
        status = AllgathervCpu<int32_t>(op_record->in_tensor,
                                        absl::Span<int32_t>(first_rank_sizes.data(), first_rank_sizes.size()),
                                        op_record->out_tensor);
        break;
      case message::DataType_TF_FLOAT32:
        status = AllgathervCpu<float>(op_record->in_tensor,
                                      absl::Span<int32_t>(first_rank_sizes.data(), first_rank_sizes.size()),
                                      op_record->out_tensor);
        break;
    }

    MPI_LOG << "#rank" << mpi_rank() << " Finished collective op";
  } else {
    LOG(FATAL) << "Not expected response type";
  }

  CHECK(op_record->out_tensor);

  if (status.ok()) {
    op_record->callback(*op_record->out_tensor);
  } else {
    op_record->callback(status);
  }
}

void BackgroundThreadLoop() {
  auto& message_queue = CollectiveState::Global().message_queue;
  auto& message_table = CollectiveState::Global().message_table;
  auto& tensor_table  = CollectiveState::Global().tensor_table;

  if (IsCoordinator()) {
    message_table.reset(new MessageTable);
  }

  std::queue<std::string> ready_to_reduce;

  // We treat the all the nodes as workers, the rank 0 is the coordinator. When a worker launched Allreduce on a
  // tensor, it just push a RequestMessage(marked as type ALLREDUCE) to the message_queue, record a
  // CommunicationDoneCallback. A background thread will process the messages in the message_queue and send the
  // RequestMessage to coordinator. When a coordinator get a ALLREDUCE RequestMessage, it will put a record in the
  // MessageTable, where a counter keep recording the latest count of RequestMessage on a tensor. When all the
  // workers' ALLREDUCE message on a specific tensor arrived on coordinator, the coordinator will schedule a actual
  // Allreduce across all the nodes with the following steps:
  //   1. The coordinator send ResponseMessages to all the workers with the tensor's name.
  //   2. The background thread process the message_queue and get the message, do Allreduce at once
  //   3. When a tensor's Allreduce is done, the background thread call the CommunicationDoneCallback, and that will
  //   continue the running of the following TensorFlow Ops.

  // This is used by the coordinator to record the requests from workers and send response.
  std::unordered_map<std::string, std::vector<RpcMsgHead>> tensor_request_msg_heads;

  auto* collective_service = RpcServer::Global().LookupService(kCoordinatorRpcServiceName);
  CHECK(collective_service);

  do {
    // process all the message in the queue
    std::pair<RpcMsgHead, RequestMessage> message;
    // The rank 0's RequestMessage is pushed to the message_queue directlly, the other workers' RequestMessages send
    // to the coordinator's message_queue by RPC service.
    if (message_queue->Read(&message)) {  // this will hang if message_queue is empty
      MPI_LOG << " read a message from message_queue";
      if (!IsCoordinator()) {  // send message to coordinator
        MPI_LOG << " send a request to coordinator";
        // callback: When the response arrived from coordinator, a Allreduce will be performed at worker.
        RpcCallback callback = [&](RpcMsgHead head, uint8_t* buf) {
          auto msg                  = flatbuffers::GetRoot<message::ResponseMessage>(buf);
          auto response_type        = msg->response_type();
          auto tensor_name          = msg->tensor_name()->str();
          std::string error_message = msg->error_message() ? msg->error_message()->str() : "";

          // NOTE Shutdown is passed by request, so this branch is disabled currentlly.
          if (response_type == message::ResponseType_SHUTDOWN) {
            CollectiveState::Global().shut_down = true;
            return;
          }

          if (response_type == message::ResponseType_ERROR) {
            auto status = tensorflow::errors::FailedPrecondition(error_message);
            tensor_table[tensor_name].callback(status);
            return;
          }

          if (response_type == message::ResponseType_ALLGATHER) {
            tensor_table[tensor_name].sizes_vec.assign(msg->tensor_sizes()->begin(), msg->tensor_sizes()->end());
          }

          CHECK(response_type == message::ResponseType_ALLGATHER || response_type == message::ResponseType_ALLREDUCE ||
                response_type == message::ResponseType_BROADCAST)
              << "Unsupported response_type found: " << response_type;

          OpRecord* op_record = CollectiveState::Global().LookupOpRecordGuarded(tensor_name);
          CHECK(op_record);
          PerformCollectiveOp(op_record, response_type, tensor_name, error_message);

          CHECK(op_record->callback);
          op_record->callback(*op_record->out_tensor);

          CollectiveState::Global().EraseOpRecordGuarded(tensor_name);
        };

        // The workers other than rank 0 send its RequestMessage to the coordinator.
        RpcServer::Global().SendRequest(
            0, collective_service, message.second.GetBuffer(), message.second.GetSize(), callback);
      } else {
        MPI_LOG << "coordinator process the message";
        // The coordinator collect all the ready_to_reduce tensors.
        auto tensor_name = message.second.msg().tensor_name()->str();
        if (message.first.initialized())
          tensor_request_msg_heads[tensor_name].push_back(
              message.first);  // record RpcMsgHead for latter response construction.

        CHECK(message.second.HasData()) << "Message is invalid";
        bool reduce = IncreTensorCount(*message_table, std::move(message.second), mpi_size());
        if (reduce) {
          MPI_LOG << "get a ready to reduce tensor: [" << tensor_name << "]";
          ready_to_reduce.push(tensor_name);
        }
      }
    } else {
      MPI_LOG << " message channel is closed";
    }

    for (int i = 0; i < ready_to_reduce.size() && IsCoordinator(); i++) {
      auto name = ready_to_reduce.front();
      ready_to_reduce.pop();
      MPI_LOG << "coordinator process a ready tensor [" << name << "]";

      ResponseMessage response = ConstructResponseMessage(*message_table, name);
      MPI_LOG << "response.name: " << response.msg().tensor_name()->str();
      RpcMsgHead head;
      head.message_type = RpcMsgType::RESPONSE;
      head.client_id    = i;
      head.server_id    = 0;
      // TODO send the response back to the workers other than coordinator
      for (auto& head : tensor_request_msg_heads[name]) {
        MPI_LOG << "coordinator send response to " << head.client_id;
        RpcServer::Global().SendResponse(head, response.GetBuffer(), response.GetSize());
      }

      std::string error_message = response.msg().error_message() ? response.msg().error_message()->str() : "";

      std::stringstream error_stream;
      bool error{};
      MessageTable::iterator requests_iter;
      {
        std::lock_guard<std::mutex> lock(CollectiveState::Global().mu);
        requests_iter = message_table->find(name);
      }
      CHECK(requests_iter != message_table->end()) << "message table has no record called " << name;

      if (response.msg().response_type() == message::ResponseType_ALLGATHER) {
        auto first_rank_sizes =
            GatherFirstRankSizes(absl::Span<RequestMessage>(requests_iter->second.data(), requests_iter->second.size()),
                                 error,
                                 error_stream);
        tensor_table[name].sizes_vec = first_rank_sizes;
        CHECK(!error) << error_stream.str();
      }

      tensor_request_msg_heads[name].clear();

      auto* op_record = CollectiveState::Global().LookupOpRecordGuarded(name);
      CHECK(op_record);
      PerformCollectiveOp(
          op_record, response.msg().response_type(), response.msg().tensor_name()->str(), error_message);

      // Clear all the requests for this tensor.
      message_table->at(name).clear();
    }

  } while (!CollectiveState::Global().shut_down);

  MPI_LOG << " background thread stop working";
}

bool CollectiveState::initialized() const { return message_table.get(); }

void CollectiveState::Initialize() {
  // the collective_coordinator service received the RequestMessage from workers and push the message to message_queue
  // directlly. We don't process the message in this RPC service to avoid affecting the global RPC performance, for it
  // is shared by all the tasks.
  RpcServer::Global().AddService(kCoordinatorRpcServiceName, [](RpcMsgHead head, uint8_t* buf) {
    auto msg = flatbuffers::GetRoot<message::RequestMessage>(buf);
    // Copy the message.
    RequestMessage message = ::tips::collective::CreateRequestMessage(
        msg->request_rank(),
        msg->request_type(),
        msg->tensor_type(),
        msg->tensor_name()->str(),
        std::vector<int64_t>(msg->tensor_shape()->begin(), msg->tensor_shape()->end()));
    CollectiveState::Global().message_queue->WriteMove(std::make_pair(std::move(head), std::move(message)));
  });

  RpcServer::Global().AddService(kShutdownRpcServiceName, [](RpcMsgHead head, uint8_t* buf) {
    CollectiveState::Global().shut_down = true;
    mpi_barrier();
    MPI_LOG << " set global shutdown state";
    RpcServer::Global().SendResponse(head, nullptr, 0);
  });

  background_thread = std::thread([] { BackgroundThreadLoop(); });
}

void CollectiveState::Finalize() {
  MPI_LOG << " to shutdown the background threads";
  // ShutdownBackgroundService();
  MPI_LOG << " done shutdown the background threads";

  MPI_LOG << " closing the message queue";
  shut_down = true;
  message_queue->Close();
  CollectiveState::Global().background_thread.join();
}

OpRecord* CollectiveState::LookupOpRecordGuarded(const std::string& tensor_name) {
  TensorTable ::iterator it;
  {
    std::lock_guard<std::mutex> lock(mu);
    it = tensor_table.find(tensor_name);
  }
  if (it != tensor_table.end()) return &it->second;
  return nullptr;
}

bool CollectiveState::EraseOpRecordGuarded(const std::string& tensor_name) {
  std::lock_guard<std::mutex> lock(mu);
  auto it = tensor_table.find(tensor_name);
  if (it != tensor_table.end()) {
    tensor_table.erase(it);
    return true;
  }
  return false;
}

void ShutdownBackgroundService() {
  auto* shutdown_service = RpcServer::Global().LookupService(kShutdownRpcServiceName);
  CHECK(shutdown_service);
  // coordinator send shutdown message to all the processes.
  if (mpi_rank() == 0) {
    for (int serverid = 0; serverid < mpi_size(); serverid++) {
      MPI_LOG << " send shutdown request to " << serverid;
      RpcServer::Global().SendRequest(
          serverid, shutdown_service, nullptr, 0, [serverid](RpcMsgHead head, uint8_t* buf) {
            MPI_LOG << " done send shutdown request to " << serverid;
          });
    }
  }
}

}  // namespace collective
}  // namespace tips