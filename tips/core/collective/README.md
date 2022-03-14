# Design

## Concepts
There are several concepts for the collective operations during the distribute ML supported by TIPS.

- `Coordinator`: the unique special node that helps to schedule the collective operations across all the nodes,
- `BackgroudThreadLoop`: The background thread singleton that helps process the request message that sent from the local nodes,
    - It will send the messages to coordinator periodically,
- `CollectiveOp`: Represent the collective operation that is one of ALLREDUCE, ALLGATHER, BROADCAST, reference the `RequestType` for more details,
- `OpRecord`: The collective request details for a local tensor, once a tensor is ready to do collective operation, it send an OpRecord to the job queue of `BackgroundThreadLoop`,

## The overall logic
1. The task is sent to MPI, each MPI process contains one or multiple job thread(TF jobs), and a node is assigned as Coordinator node,
2. The `BackgroundThreadLoop` is launched locally(as a process-wise singleton), it contains a job queue for the collective operation requests from local threads,
3. Each time a job thread finished a gradient tensor, it sends a `OpRecord` message to the `BackgroundThreadLoop`, the `BackgroundThreadLoop` will send the messages to the coordinator node,
  a. The coordinator node holds a dictionary of all the gradient tensors, each is attached to a counter,
  b. When a tensor is ready to perform collective operation, the coordinator will launch the operation (by NCCL).
  c. Once the collective operation is finished, the coordinator will send message to all the nodes in the system to do their `callback`.
4. Repeat until the overall work is finised.
