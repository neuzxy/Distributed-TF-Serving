## Distributed TF-Serving RPC Client Demo For CTR Prediction

Speedup inference task by TF-Serving
- Rpc client split candidates to multiple RPC servers, each server process a part of inference task.
- Using batching prediction provided by TF Serving to reduce call counts of API.


Feature
- RPC Client connects to multiple RPC server, serialize by protobuf, using batching prediction provided by TF Serving to speed up inference.  
- Using Java8 CompletableFuture for async callback and ThreadPool for client thread management.
