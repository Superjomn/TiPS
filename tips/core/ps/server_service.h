#pragma once
#include "tips/core/common/flatbuffers_utils.h"
#include "tips/core/common/naive_rpc.h"
#include "tips/core/message/ps_messages_generated.h"

namespace tips {

class ServerService {
 public:
 private:
  /**
   * The server-side service that process the PUSH and PULL requests.
   */
  void AddPsService(RpcServer* server) {
    auto callback = [server, this](ZmqMessage&& zmq_msg) {
      const RpcMsgHead* msg_head = GetMsgHead(zmq_msg);
      const void* msg_content    = GetMsgContent(zmq_msg);
      CHECK(msg_head);
      CHECK(msg_content);

      if (msg_head->is_request()) {
        RpcMsgHead rsp_head;
        rsp_head.server_id = mpi_rank();
        rsp_head.client_id = msg_head->client_id;
        rsp_head.service   = msg_head->service;
        rsp_head.request   = msg_head->request;

        // parse the content
        if (msg_head->is_ps_pull()) {
          auto pull_request = GetFbsData<ps::message::PullRequest>(msg_content);
          ProceedPull(*pull_request);
        } else if (msg_head->is_ps_push()) {
          auto push_request = GetFbsData<ps::message::PushRequest>(msg_content);
          ProceedPush(*push_request);
        }
      }
    };
  }

  /**
   * Do Pull request.
   * 1. It will route the pull-message to the target table service in-process.
   * 2. Add a complete-callback handler and send back the result once the table service send back the result.
   */
  void ProceedPull(const ps::message::PullRequest& pull_request) { pull_request.keys(); }

  /**
   * Do Push request.
   * It will route the push-message to the target table service in-process.
   */
  void ProceedPush(const ps::message::PushRequest& push_request) {}
};

}  // namespace tips
