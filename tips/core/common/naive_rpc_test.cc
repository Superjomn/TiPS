#include "tips/core/common/naive_rpc.h"

#include <mpi.h>

#include <chrono>
#include <thread>

#include "tips/core/collective/utils.h"
#include "tips/core/message/test0_generated.h"

namespace tips {

using namespace std::chrono_literals;
using namespace test::test_message0;

void TestRpc(RpcServer& server) {
  RpcCallback callback = [&server](ZmqMessage&& zmq_msg) {
    const auto* head = GetMsgHead(zmq_msg);
    auto* buffer     = GetMsgContent(zmq_msg);

    CHECK(head);
    CHECK(buffer);

    std::this_thread::sleep_for(500ms);
    if (head->message_type == RpcMsgType::REQUEST) {
      LOG(INFO) << "server " << mpi_rank() << " get a request";

      RpcMsgHead response_head;
      response_head.server_id = mpi_rank();
      response_head.client_id = head->client_id;
      response_head.service   = head->service;
      response_head.request   = head->request;

      auto msg = flatbuffers::GetRoot<MessageRequest>(buffer);

      int v             = msg->v();
      std::string greet = msg->greet()->str();

      CHECK_EQ(greet, "hello node" + std::to_string(v));
      LOG(INFO) << mpi_rank() << " get message from master: " << greet;

      {
        FlatBufferBuilder builder;
        auto msg = MessageResponseBuilder(builder);
        msg.add_from_rank(mpi_rank());
        builder.Finish(msg.Finish());

        server.SendResponse(response_head, builder.GetBufferPointer(), builder.GetSize());
      }
    }

    if (head->message_type == RpcMsgType::RESPONSE) {
      LOG(INFO) << "server " << mpi_rank() << " get a response";
    }
  };

  // CHECK_EQ(mpi_size(), 3);
  auto* service = server.AddService("test", callback);

  LOG(INFO) << "to Intialize server";
  server.Initialize();
  mpi_barrier();

  if (mpi_rank() == 0) {
    RpcCallback callback = [&server](ZmqMessage&& zmq_msg) {};

    LOG(INFO) << "master send request...";
    {
      FlatBufferBuilder builder;
      auto greet = builder.CreateString("hello node1");
      auto msg   = MessageRequestBuilder(builder);
      msg.add_greet(greet);
      msg.add_v(1);
      builder.Finish(msg.Finish());
      server.SendRequest(0, service, builder.GetBufferPointer(), builder.GetSize(), callback);
    }
    {
      FlatBufferBuilder builder;
      auto greet = builder.CreateString("hello node2");
      auto msg   = MessageRequestBuilder(builder);
      msg.add_greet(greet);
      msg.add_v(2);
      builder.Finish(msg.Finish());
      server.SendRequest(0, service, builder.GetBufferPointer(), builder.GetSize(), callback);
    }
  }

  mpi_barrier();
  std::this_thread::sleep_for(2000ms);
  server.Finalize();
}

}  // namespace tips

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  tips::RpcServer server;

  tips::TestRpc(server);

  MPI_Finalize();
  return 0;
}
