#include "tips/core/common/zmq_message.h"

#include "tips/core/common/naive_rpc.h"

namespace tips {

const RpcMsgHead* GetMsgHead(const ZmqMessage& msg) {
  if (msg.length() == 0) return nullptr;
  return reinterpret_cast<const RpcMsgHead*>(msg.buffer());
}

const void* GetMsgContent(const ZmqMessage& msg) {
  CHECK_GE(msg.length(), sizeof(RpcMsgHead));
  if (msg.length() == sizeof(RpcMsgHead)) return nullptr;
  return msg.buffer() + sizeof(RpcMsgHead);
}

}  // namespace tips