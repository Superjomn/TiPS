#include "swiftps/core/common/swifts_rpc.h"

#include <mpi.h>

#include <chrono>
#include <thread>

namespace swifts {

using namespace std::chrono_literals;

void TestRpc() {
  RpcServer server;
  RpcCallback callback = [&server](const RpcMsgHead& head, NaiveBuffer& buffer) {
    std::this_thread::sleep_for(500ms);
    if (head.message_type == RpcMsgType::REQUEST) {
      LOG(INFO) << "server " << mpi_rank() << " get a request";

      RpcMsgHead response_head;
      response_head.server_id = mpi_rank();
      response_head.client_id = head.client_id;
      response_head.service   = head.service;
      response_head.request   = head.request;

      NaiveBuffer write_buf;
      write_buf << 123;

      server.SendResponse(response_head, &write_buf, 1);
    }
    if (head.message_type == RpcMsgType::RESPONSE) {
      LOG(INFO) << "server " << mpi_rank() << " get a response";
    }
  };

  CHECK_EQ(mpi_size(), 3);
  LOG(INFO) << "rank: " << mpi_rank();
  auto* service = server.AddService(callback);

  LOG(INFO) << "to Intialize server";
  server.Initialize();
  mpi_barrier();
  LOG(INFO) << "server initialized";

  if (mpi_rank() == 0) {
    NaiveBuffer writebuf;
    writebuf << 123;

    RpcCallback callback = [&server](RpcMsgHead head, NaiveBuffer& buf) {
      LOG(INFO) << "master get head " << head.message_type;
    };

    LOG(INFO) << "master send request...";
    server.SendRequest(1, service, &writebuf, 1, callback);
    server.SendRequest(2, service, &writebuf, 1, callback);

    NaiveBuffer emptybuf;
    server.SendRequest(1, service, &emptybuf, 1, callback);
    server.SendRequest(2, service, &emptybuf, 1, callback);
  }

  mpi_barrier();
  std::this_thread::sleep_for(2000ms);
  server.Finalize();

  LOG(INFO) << "final";
}

}  // namespace swifts

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  swifts::TestRpc();

  MPI_Finalize();
  return 0;
}
