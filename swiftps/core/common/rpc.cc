#include "swiftps/core/common/rpc.h"

namespace swifts {

RpcServer::~RpcServer() {
  for (auto* x : services_) {
    delete x;
  }
}

RpcService* RpcServer::AddService(RpcCallback callback) {
  auto* new_service = new RpcService(std::move(callback));
  services_.insert(new_service);
  return new_service;
}
}  // namespace swifts
