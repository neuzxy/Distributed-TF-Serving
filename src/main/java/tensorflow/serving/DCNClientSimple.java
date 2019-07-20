package tensorflow.serving;

import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import io.grpc.netty.GrpcSslContexts;
import io.grpc.netty.NettyChannelBuilder;
import io.netty.handler.ssl.util.InsecureTrustManagerFactory;
import org.tensorflow.framework.DataType;
import org.tensorflow.framework.TensorProto;
import org.tensorflow.framework.TensorShapeProto;

import javax.net.ssl.SSLException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.LongStream;

public class DCNClientSimple {

    private static final int FIELD_NUM = 43;
    private static final int candidateNum = 1500;
    private static final int requestNum = 100;

    public static void main(String[] args) {
                Predict.PredictRequest.Builder request = Predict.PredictRequest.newBuilder();
        Model.ModelSpec.Builder modelSpec = Model.ModelSpec.newBuilder();
        modelSpec.setName("DCN");
        modelSpec.setSignatureName("serving_default");
        request.setModelSpec(modelSpec);


        TensorShapeProto.Builder tensorShapeBuilder = TensorShapeProto.newBuilder();
        tensorShapeBuilder.addDim(TensorShapeProto.Dim.newBuilder().setSize(candidateNum));
        tensorShapeBuilder.addDim(TensorShapeProto.Dim.newBuilder().setSize(FIELD_NUM));
        TensorShapeProto tensorShapeProto = tensorShapeBuilder.build();


        TensorProto.Builder featIdsBuilder = TensorProto.newBuilder();
        featIdsBuilder.setTensorShape(tensorShapeProto);
        featIdsBuilder.setDtype(DataType.DT_INT64);
        featIdsBuilder.addAllInt64Val(LongStream.range(1, FIELD_NUM + 1).boxed().collect(Collectors.toList()));
        featIdsBuilder.addAllInt64Val(LongStream.range(FIELD_NUM, 2 * FIELD_NUM + 1).boxed().collect(Collectors.toList()));
        request.putInputs("feat_ids", featIdsBuilder.build());


        TensorProto.Builder featWtsBuilder = TensorProto.newBuilder();
        featWtsBuilder.setTensorShape(tensorShapeProto);
        featWtsBuilder.setDtype(DataType.DT_FLOAT);
        featWtsBuilder.addAllFloatVal(new ArrayList<Float>(Collections.nCopies(FIELD_NUM, 1F)));
        featWtsBuilder.addAllFloatVal(new ArrayList<Float>(Collections.nCopies(FIELD_NUM, 1F)));
        request.putInputs("feat_wts", featWtsBuilder.build());

        ManagedChannel channel = ManagedChannelBuilder.forAddress("127.0.0.1", 9999).usePlaintext().build();

        PredictionServiceGrpc.PredictionServiceBlockingStub stub = PredictionServiceGrpc.newBlockingStub(channel);


        Predict.PredictResponse response = stub.predict(request.build());
        System.out.println(response);
    }
}
