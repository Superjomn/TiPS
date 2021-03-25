#include "tips/core/collective/coordinator.h"

namespace tips {
namespace collective {

const char* kShutdownRpcServiceName    = "collective_shutdown";
const char* kCoordinatorRpcServiceName = "collective_coordinator";

bool IncreTensorCount(MessageTable& table, RequestMessage&& msg, int mpi_size) {
  CHECK(msg.HasData()) << "Message is invalid";
  auto name = msg.msg().tensor_name()->str();
  LOG(INFO) << "IncreTensorCount: " << msg.msg().tensor_name()->str();
  auto table_iter = table.find(name);
  if (table_iter == table.end()) {
    table[name].emplace_back(std::move(msg));
    table_iter = table.find(name);
  } else {
    table_iter->second.emplace_back(std::move(msg));
  }

  LOG(INFO) << "tensor record " << table_iter->first << " count: " << table_iter->second.size();
  for (int i = 0; i < table_iter->second.size(); i++) {
    CHECK(table_iter->second[i].HasData()) << i << "-th Message is invalid";
  }

  return table_iter->second.size() == mpi_size;
}

ResponseMessage ConstructResponseMessage(MessageTable& table, const std::string& name) {
  LOG(INFO) << "constructing the response message";
  auto it = table.find(name);
  CHECK(it != table.end()) << "message table doesn't have item called " << name;

  const auto& requests = it->second;
  CHECK_GT(requests.size(), 0);
  LOG(INFO) << "requests.size: " << requests.size();
  LOG(INFO) << "to visit request";
  CHECK(requests[0].HasData()) << "request message is invalid";
  LOG(INFO) << requests[0].msg().tensor_name()->str();

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
  LOG(INFO) << "message.name: " << message.msg().tensor_name()->str();

  LOG(INFO) << "#rank-" << mpi_rank() << " enqueue record [" << record.name << "] to the queue";
  CHECK(record.callback);

  {
    std::lock_guard<std::mutex> lock(CollectiveState::Global().mu);
    CollectiveState::Global().tensor_table.emplace(record.name, record);
  }

  // TODO(Superjomn) avoid creating RpcMsgHead here.
  RpcMsgHead head;
  CollectiveState::Global().message_queue->WriteMove(std::make_pair(head, std::move(message)));
}

void PerformCollectiveOp(TensorTable& tensor_table,
                         message::ResponseType response_type,
                         const std::string name,
                         const std::string& error_msg) {
  LOG(INFO) << "Perform collective!";
  OpRecord record;
  {
    std::lock_guard<std::mutex> lock(CollectiveState::Global().mu);

    auto it = tensor_table.find(name);
    CHECK(it != tensor_table.end());

    CHECK(response_type == message::ResponseType_ALLREDUCE || response_type == message::ResponseType_ALLGATHER ||
          response_type == message::ResponseType_ERROR);

    record = it->second;

    // tensor_table.erase(it);
  }

  Status status;
  auto dtype = record.dtype;

  if (response_type == message::ResponseType_ALLREDUCE) {
    LOG(INFO) << "Run MPI ALLREDUCE ...";
    switch (dtype) {
      case message::DataType_TF_INT32:
        LOG(INFO) << "Allreduce int32 ...";
        status = AllreduceCpu<int32_t>(record.in_tensor, record.out_tensor, CollectiveOpKind::SUM);
        break;
      case message::DataType_TF_FLOAT32:
        LOG(INFO) << "Allreduce float32 ...";
        status = AllreduceCpu<float>(record.in_tensor, record.out_tensor, CollectiveOpKind::SUM);
        break;
    }
  } else if (response_type == message::ResponseType_ALLGATHER) {
    LOG(INFO) << "Run MPI ALLGATHER ...";
    switch (dtype) {
      switch (dtype) {
        case message::DataType_TF_INT32:
          status = AllgatherCpu<int32_t>(record.in_tensor, record.out_tensor);
          break;
        case message::DataType_TF_FLOAT32:
          status = AllgatherCpu<float>(record.in_tensor, record.out_tensor);
          break;
      }
    }
  } else if (response_type == message::ResponseType_ERROR) {
    status = tensorflow::errors::FailedPrecondition(error_msg);
  }

  if (status.ok()) {
    record.callback(StatusOr<Tensor>(*record.out_tensor));
  } else {
    record.callback(StatusOr<Tensor>(status));
  }
}

void BackgroundThreadLoop() {
  auto& message_queue = CollectiveState::Global().message_queue;
  auto& message_table = CollectiveState::Global().message_table;

  if (IsCoordinator()) {
    CollectiveState::Global().message_table.reset(new MessageTable);
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
  std::unordered_map<std::string, std::vector<RpcMsgHead>> tensor_requests;

  auto* rpc_service = RpcServer::Global().LookupService(kCoordinatorRpcServiceName);
  CHECK(rpc_service);

  do {
    // process all the message in the queue
    std::pair<RpcMsgHead, RequestMessage> message;
    // The rank 0's RequestMessage is pushed to the message_queue directlly, the other workers' RequestMessages send
    // to the coordinator's message_queue by RPC service.
    if (message_queue->Read(&message)) {  // this will hang if message_queue is empty
      LOG(INFO) << "rank-" << mpi_rank() << " read a message from message_queue";
      if (!IsCoordinator()) {  // send message to coordinator
        LOG(INFO) << "rank-" << mpi_rank() << " send a request to coordinator";
        // callback: When the response arrived from coordinator, a Allreduce will be performed at worker.
        RpcCallback callback = [&](RpcMsgHead head, uint8_t* buf) {
          auto msg = flatbuffers::GetRoot<message::ResponseMessage>(buf);
          if (msg->response_type() == message::ResponseType_SHUTDOWN) {
            CollectiveState::Global().shut_down = true;
          }

          if (msg->response_type() == message::ResponseType_ALLREDUCE) {
            LOG(INFO) << "#rank-" << mpi_rank() << " get response for [" << msg->tensor_name()->str() << "]";
            PerformCollectiveOp(CollectiveState::Global().tensor_table,
                                msg->response_type(),
                                msg->tensor_name()->str(),
                                msg->error_message() ? msg->error_message()->str() : "");

            auto record_iter = CollectiveState::Global().tensor_table.find(msg->tensor_name()->str());
            CHECK(record_iter != CollectiveState::Global().tensor_table.end())
                << "tensor_table has no record for [" << msg->tensor_name()->str() << "]";
            auto& op_record = record_iter->second;
            CHECK(op_record.callback);
            op_record.callback(StatusOr<Tensor>(*op_record.out_tensor));

            CollectiveState::Global().tensor_table.erase(record_iter);
          } else {
            LOG(FATAL) << "Not supported response type: " << msg->response_type();
          }
        };
        // The workers other than rank 0 send its RequestMessage to the coordinator.
        RpcServer::Global().SendRequest(0, rpc_service, message.second.GetBuffer(), message.second.GetSize(), callback);
      } else {
        LOG(INFO) << "coordinator process the message";
        // The coordinator collect all the ready_to_reduce tensors.
        auto tensor_name = message.second.msg().tensor_name()->str();
        if (message.first.initialized())
          tensor_requests[tensor_name].push_back(message.first);  // record RpcMsgHead for latter response construction.

        CHECK(message.second.HasData()) << "Message is invalid";
        bool reduce = IncreTensorCount(*message_table, std::move(message.second), mpi_size());
        if (reduce) {
          LOG(INFO) << "get a ready to reduce tensor: [" << tensor_name << "]";
          ready_to_reduce.push(tensor_name);
        }
      }
    }

    if (IsCoordinator()) {
      for (int i = 0; i < ready_to_reduce.size(); i++) {
        auto name = ready_to_reduce.front();
        ready_to_reduce.pop();
        LOG(INFO) << "coordinator process a ready tensor [" << name << "]";

        ResponseMessage response = ConstructResponseMessage(*message_table, name);
        LOG(INFO) << "response.name: " << response.msg().tensor_name()->str();
        RpcMsgHead head;
        head.message_type = RpcMsgType::RESPONSE;
        head.client_id    = i;
        head.server_id    = 0;
        // TODO send the response back to the workers other than coordinator
        for (auto& head : tensor_requests[name]) {
          LOG(INFO) << "coordinator send response to " << head.client_id;
          RpcServer::Global().SendResponse(head, response.GetBuffer(), response.GetSize());
        }
        tensor_requests[name].clear();

        std::string error_message = response.msg().error_message() ? response.msg().error_message()->str() : "";
        PerformCollectiveOp(CollectiveState::Global().tensor_table,
                            response.msg().response_type(),
                            response.msg().tensor_name()->str(),
                            error_message);
      }
    }

  } while (!CollectiveState::Global().shut_down);
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
    RpcServer::Global().SendResponse(head, nullptr, 0);
  });

  background_thread = std::thread([] { BackgroundThreadLoop(); });
}

void CollectiveState::Finalize() {
  LOG(INFO) << "Shutdown the background threads";
  ShutdownBackgroundService();

  if (IsCoordinator()) {
    LOG(INFO) << "Close the message queue";
    message_queue->Close();
  }

  LOG(INFO) << "Join the background thread";
  message_queue->Close();
  shut_down = true;
  CollectiveState::Global().background_thread.join();
}

void ShutdownBackgroundService() {
  auto* shutdown_service = RpcServer::Global().LookupService(kShutdownRpcServiceName);
  CHECK(shutdown_service);
  // coordinator send shutdown message to all the processes.
  if (mpi_rank() == 0) {
    for (int serverid = 0; serverid < mpi_size(); serverid++) {
      LOG(INFO) << "Send shutdown request to " << serverid;
      RpcServer::Global().SendRequest(
          serverid, shutdown_service, nullptr, 0, [serverid](RpcMsgHead head, uint8_t* buf) {
            VLOG(2) << "Done send shutdown request to " << serverid;
          });
    }
  }

  mpi_barrier();
}

}  // namespace collective
}  // namespace tips