package tensorflow.serving;

import com.sun.org.apache.xpath.internal.operations.Mod;
import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import io.grpc.netty.GrpcSslContexts;
import io.grpc.netty.NettyChannelBuilder;
import io.netty.handler.ssl.util.InsecureTrustManagerFactory;
import org.omg.PortableServer.LIFESPAN_POLICY_ID;
import org.tensorflow.framework.DataType;
import org.tensorflow.framework.TensorProto;
import org.tensorflow.framework.TensorShapeProto;

import javax.net.ssl.SSLException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.*;
import java.util.stream.Collectors;
import java.util.stream.LongStream;

public class DCNClient {

    private static final int FIELD_NUM = 43;
    
    private static final boolean isFullAsyncMode = true; 
    private static final int port = 9999;
    private static final int candidateNum = 1500;
    private static final int requestNum = 1000;
    private static final int concurrentNum = 6;

    private static final String modelName = "DCN";
    private static final String modelSignature = "serving_default";
    private static final String outputKey = "prediction_node";
    

    private static final List<String> hostsList = Arrays.asList("9.91.21.26", "9.91.21.28", "9.91.21.29");

//    private static final List<String> hostsList = Arrays.asList("9.91.21.26", "9.91.21.28", "10.154.236.199");

    private static ExecutorService threadPool = Executors.newFixedThreadPool(16);

    private static List<Double> timeLists = Collections.synchronizedList(new ArrayList<>());

    private static <T> List<List<T>> partitionList(List<T> list, final int partNum) {
        List<List<T>> parts = new ArrayList<>();
        final int N = list.size();
        int range = N / partNum;
        for (int i = 0; i < partNum - 1; i++) {
            parts.add(list.subList(i * range, (i + 1) * range));
        }
        parts.add(list.subList((partNum - 1) * range, N));
        return parts;
    }

    private static List<List<Long>> getFeatureIds() {
        List<Long> featureIdsList = new ArrayList<>();
        ArrayList<Long> list = LongStream.range(1, FIELD_NUM + 1).boxed().collect(Collectors
                .toCollection(ArrayList::new));
        for (int i = 0; i < candidateNum; i++) {
            featureIdsList.addAll(list);
        }
        return partitionList(featureIdsList, hostsList.size());
    }

    private static List<List<Float>> getFeatureWts() {
        List<Float> featureWtsList = new ArrayList<>();
        ArrayList<Float> list = new ArrayList<Float>(Collections.nCopies(FIELD_NUM, 1F));
        for (int i = 0; i < candidateNum; i++) {
            featureWtsList.addAll(list);
        }
        return partitionList(featureWtsList, hostsList.size());
    }

    private static TensorShapeProto getTensorShapeProto(int dim) {
        TensorShapeProto.Builder tensorShapeBuilder = TensorShapeProto.newBuilder();
        tensorShapeBuilder.addDim(TensorShapeProto.Dim.newBuilder().setSize(dim));
        tensorShapeBuilder.addDim(TensorShapeProto.Dim.newBuilder().setSize(FIELD_NUM));
        return tensorShapeBuilder.build();
    }

    private static Model.ModelSpec.Builder getModelSpec(String modelName, String modelSiganure) {
        Model.ModelSpec.Builder modelSpec = Model.ModelSpec.newBuilder();
        modelSpec.setName(modelName);
        modelSpec.setSignatureName(modelSiganure);
        return modelSpec;
    }


    private static Predict.PredictResponse sendRequest(ManagedChannel channel, List<Long> featureIds,
                                                       List<Float> featureWts, Model.ModelSpec.Builder modelSpec) {

        Predict.PredictRequest.Builder request = Predict.PredictRequest.newBuilder();
        request.setModelSpec(modelSpec);

        TensorShapeProto tensorShapeProto = getTensorShapeProto(featureIds.size() / FIELD_NUM);
        TensorProto.Builder featIdsBuilder = TensorProto.newBuilder();
        featIdsBuilder.setTensorShape(tensorShapeProto);
        featIdsBuilder.setDtype(DataType.DT_INT64);
        featIdsBuilder.addAllInt64Val(featureIds);
        request.putInputs("feat_ids", featIdsBuilder.build());

        TensorProto.Builder featWtsBuilder = TensorProto.newBuilder();
        featWtsBuilder.setTensorShape(tensorShapeProto);
        featWtsBuilder.setDtype(DataType.DT_FLOAT);
        featWtsBuilder.addAllFloatVal(featureWts);
        request.putInputs("feat_wts", featWtsBuilder.build());


        PredictionServiceGrpc.PredictionServiceBlockingStub stub = PredictionServiceGrpc.newBlockingStub(channel);
        Predict.PredictResponse response = stub.predict(request.build());

        return response;
    }


    private static List<ManagedChannel> getChannels() {
        List<ManagedChannel> channelList = new ArrayList<>();
        for (String host: hostsList) {
            ManagedChannel channel = ManagedChannelBuilder.forAddress(host, port).usePlaintext().build();
            channelList.add(channel);
        }
        return channelList;
    }

    private static void shutDownChannels(List<ManagedChannel> channelList) {
        for (ManagedChannel channel: channelList) {
            try {
                channel.awaitTermination(2, TimeUnit.SECONDS);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
    }

    private static void processARequest(List<ManagedChannel> channelList,
                                        List<List<Long>> featIdsList,
                                        List<List<Float>> featWtsList,
                                        Model.ModelSpec.Builder modelSpec) {
        Long startTime = System.nanoTime();

        List<Float> fullCtrList = new ArrayList<>();
        final int taskNum = hostsList.size();

        if (isFullAsyncMode) {
            List<CompletableFuture<Predict.PredictResponse>> futureList = new ArrayList<>();
            for (int i = 0; i < taskNum; i++) {
                final int idx = i;
                futureList.add(
                        CompletableFuture.supplyAsync(
                                () -> sendRequest(channelList.get(idx), featIdsList.get(idx), featWtsList.get(idx), modelSpec),
                                threadPool
                        )
                );
            }

            List<Predict.PredictResponse> responseList =
                    futureList.stream().map(CompletableFuture::join).collect(Collectors.toList());

            for (int i = 0; i < taskNum; i++) {
                List<Float> ctrList = responseList.get(i).getOutputsMap().get(outputKey).getFloatValList();
                fullCtrList.addAll(ctrList);
            }
        } else {
           Callable[] tasks = new Callable[taskNum];
           Future[] futures = new Future[taskNum];
           for (int i = 0; i < taskNum; i++) {
               final int idx = i;
               tasks[idx] = () -> sendRequest(channelList.get(idx), featIdsList.get(idx), featWtsList.get(idx), modelSpec);
               futures[idx] = threadPool.submit(tasks[idx]);
           }


           List<Integer> doneTaskList = new ArrayList<>();
           while (doneTaskList.size() < taskNum) {
               for (int i = 0; i < taskNum; i++) {
                   if (!doneTaskList.contains(i) && futures[i].isDone()) {
                       doneTaskList.add(i);
                       Predict.PredictResponse response = null;
                       try {
                           response = (Predict.PredictResponse) futures[i].get();
                           List<Float> ctrList = response.getOutputsMap().get(outputKey).getFloatValList();
                           fullCtrList.addAll(ctrList);
                       } catch (InterruptedException | ExecutionException e) {
                           e.printStackTrace();
                           return;
                       }
                   }

               }
           }
        }

        Collections.sort(fullCtrList);


        long endTime = System.nanoTime();
        double reqTime = (endTime - startTime) / 1e6;
        timeLists.add(reqTime);
        System.out.println("Thread " + Thread.currentThread().getName()
                + ". Time cost with " + fullCtrList.size() + " is " + reqTime  + " ms");
    }

    public static void main(String[] args) {
        List<ManagedChannel> channelList = getChannels();

        Model.ModelSpec.Builder modelSpec = getModelSpec(modelName, modelSignature);
        List<List<Long>> featIdsList = getFeatureIds();
        List<List<Float>> featWtsList = getFeatureWts();

        // Multi-threads send requests
        Runnable[] tasks = new Runnable[concurrentNum];
        Thread[] threads = new Thread[concurrentNum];
        for (int i = 0; i < tasks.length; i++) {
             tasks[i] = () -> {
                for (int j = 0; j < requestNum; j++) {
//                    System.out.println("Thread name " + Thread.currentThread().getName() + " process request " + j);
                    processARequest(channelList, featIdsList, featWtsList, modelSpec);
                }
            };
            threads[i] = new Thread(tasks[i]);
            threads[i].start();
        }

        for (int i = 0; i < concurrentNum; i++) {
            try {
                threads[i].join();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }

        double avgTime = timeLists.stream().mapToDouble(Double::doubleValue).sum() / timeLists.size();
        System.out.println("Average time cost with " + candidateNum + " is " + avgTime
                + " ms with " + timeLists.size() + " requests");

        threadPool.shutdown();
        shutDownChannels(channelList);

    }
}
