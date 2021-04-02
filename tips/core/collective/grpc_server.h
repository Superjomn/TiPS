#pragma once

#include "tips/core/collective/grpc_server.h"
#include "tips/core/message/collective_messages.grpc.fb.h"

namespace tips {
namespace collective {
namespace message {

class CollectiveImpl : public Collective::Service {
 public:
  ::grpc::Status RequestDo(::grpc::ServerContext* context, const RequestMessage* request, Empty* reply) override {}

  // WorkerDo(ResponseMessage): Empty;
};

}  // namespace message
}  // namespace collective
}  // namespace tips
