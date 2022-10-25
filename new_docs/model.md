Model
=====

1. [INC Model](#inc-model)
2. [Working Flow](#working-flow)
3. [Framework Model Support Matrix](#framework-model-support-matrix)
4. [Model API summary](#model-api-summary)

## INC model

The Neural Compressor Model feature is used to encapsulate the behavior of model building and saving. By simply providing information such as different model formats and framework_specific_info, Neural Compressor performs optimizations and quantization on this model object and returns an Neural Compressor Model object for further model persisting or benchmarking. An Neural Compressor Model helps users to maintain necessary model information which is needed during optimization and quantization such as the input/output names, workspace path, and other model format knowledge. This helps unify the features gap brought by different model formats and frameworks.


## Working flow

Users can create, use, and save models in the following manner.:

```python
from neural_compressor.common import Model
inc_model = Model(input_model)
```

or

```python
from neural_compressor.experimental import Quantizati on
quantizer = Quantization(args.config)
quantizer.model = model
q_model = quantizer.fit()
```


**Note**:

**Input_model**: 
 - For Tensorflow model, could be path to frozen pb file, path to ckpt or savedmodel folder, loaded estimator/graph_def/graph/keras model object.
 - For PyTorch model, it's torch.nn.model instance.
 - For ONNX model, it's a path of onnx file or an onnx model instance.
 - For MXNet model, it's mxnet.symbol.Symbol or gluon.HybirdBlock instance.


## Framework Model Support Matrix

### TensorFlow

| Model format | Parameters | Comments | Usage |
| ------ | ------ |------|------|
| frozen pb | **model**(str): path to frozen pb <br> **framework_specific_info**(dict): information about model and framework, such as input_tensor_names, input_tensor_names, workspace_path and name <br> **kwargs**(dict): other required parameters | **Examples**: <br> [../examples/tensorflow/image_recognition](../examples/tensorflow/image_recognition) <br> [../examples/tensorflow/oob_models](../examples/tensorflow/oob_models) <br> **Save format**: <br> frozen pb | from neural_compressor.experimental import Quantization, common <br> quantizer = Quantization(args.config) <br> quantizer.model = model <br> q_model = quantizer.fit() <br> **model is the path of model, like ./path/to/frozen.pb** |
| Graph object | **model**(tf.compat.v1.Graph): tf.compat.v1.Graph object  <br> **framework_specific_info**(dict): information about model and framework, such as input_tensor_names, input_tensor_names, workspace_path and name <br> **kwargs**(dict): other required parameters | **Examples**: <br> [../examples/tensorflow/style_transfer](../examples/tensorflow/style_transfer) <br> [../examples/tensorflow/recommendation/wide_deep_large_ds](../examples/tensorflow/recommendation/wide_deep_large_ds) <br> **Save format**: <br> frozen pb | from neural_compressor.experimental import Quantization, common <br> quantizer = Quantization(args.config) <br> quantizer.model = model <br> q_model = quantizer.fit() <br> **model is the object of tf.compat.v1.Graph** |
| Graph object | **model**(tf.compat.v1.GraphDef) tf.compat.v1.GraphDef object <br> **framework_specific_info**(dict): information about model and framework, such as input_tensor_names, input_tensor_names, workspace_path and name <br> **kwargs**(dict): other required parameters | **Save format**: <br> frozen pb | from neural_compressor.experimental import Quantization, common <br> quantizer = Quantization(args.config) <br> quantizer.model = model <br> q_model = quantizer.fit() <br> **model is the object of tf.compat.v1.GraphDef** |
| tf1.x checkpoint | **model**(str): path to checkpoint <br> **framework_specific_info**(dict): information about model and framework, such as input_tensor_names, input_tensor_names, workspace_path and name <br> **kwargs**(dict): other required parameters | **Examples**: <br> [../examples/helloworld/tf_example4](../examples/helloworld/tf_example4) <br> [../examples/tensorflow/object_detection](../examples/tensorflow/object_detection)  <br> **Save format**: <br> frozen pb | from neural_compressor.experimental import Quantization, common <br> quantizer = Quantization(args.config) <br> quantizer.model = model <br> q_model = quantizer.fit() <br> **model is the path of model, like ./path/to/ckpt/** |
| keras.Model object | **model**(tf.keras.Model): tf.keras.Model object <br> **framework_specific_info**(dict): information about model and framework, such as input_tensor_names, input_tensor_names, workspace_path and name <br> **kwargs**(dict): other required parameters | **Save format**: <br> keras saved model | from neural_compressor.experimental import Quantization, common <br> quantizer = Quantization(args.config) <br> quantizer.model = model <br> q_model = quantizer.fit() <br> **model is the object of tf.keras.Model** |
| keras saved model | **model**(str): path to keras saved model <br> **framework_specific_info**(dict): information about model and framework, such as input_tensor_names, input_tensor_names, workspace_path and name <br> **kwargs**(dict): other required parameters | **Examples**: <br> [../examples/helloworld/tf_example2](../examples/helloworld/tf_example2) <br> **Save format**: <br> keras saved model | from neural_compressor.experimental import Quantization, common <br> quantizer = Quantization(args.config) <br> quantizer.model = model <br> q_model = quantizer.fit() <br> **model is the path of model, like ./path/to/saved_model/** |
| tf2.x saved model | **model**(str): path to saved model <br> **framework_specific_info**(dict): information about model and framework, such as input_tensor_names, input_tensor_names, workspace_path and name <br> **kwargs**(dict): other required parameters | **Save format**: <br> saved model | from neural_compressor.experimental import Quantization, common <br> quantizer = Quantization(args.config) <br> quantizer.model = model <br> q_model = quantizer.fit() <br> **model is the path of model, like ./path/to/saved_model/** |
| tf2.x h5 format model  | | TBD | |
| slim checkpoint | **model**(str): path to slim checkpoint <br> **framework_specific_info**(dict): information about model and framework, such as input_tensor_names, input_tensor_names, workspace_path and name <br> **kwargs**(dict): other required parameters | **Examples**: <br> [../examples/helloworld/tf_example3](../examples/helloworld/tf_example3) <br> **Save format**: <br> frozen pb | from neural_compressor.experimental import Quantization, common <br> quantizer = Quantization(args.config) <br> quantizer.model = model <br> q_model = quantizer.fit() <br> **model is thepath of model, like ./path/to/model.ckpt**|
| tf1.x saved model | **model**(str): path to saved model, **framework_specific_info**(dict): information about model and framework, such as input_tensor_names, input_tensor_names, workspace_path and name <br> **kwargs**(dict): other required parameters | **Save format**: <br> saved model | from neural_compressor.experimental import Quantization, common <br> quantizer = Quantization(args.config) <br> quantizer.model = model <br> q_model = quantizer.fit() <br> **model is the path of model, like ./path/to/saved_model/** |
| tf2.x checkpoint | | Not support yes. As tf2.x checkpoint only has weight and does not contain any description of the computation, please use different tf2.x model for quantization | |

### PyTorch

| Model format | Parameters | Comments | Usage |
| ------ | ------ |------|------|
| torch.nn.model | **model**(torch.nn.model): torch.nn.model object <br> **framework_specific_info**(dict): information about model and framework <br> **kwargs**(dict): other required parameters | **Save format**: <br> Without Intel PyTorch Extension(IPEX): /save_path/best_model.pt <br> With IPEX: /save_path/best_configure.json | from neural_compressor.experimental import Quantization, common <br> quantizer = Quantization(args.config) <br> quantizer.model = model <br> q_model = quantizer.fit() <br> **model is a torch.nn.model object** |

### ONNX

| Model format | Parameters | Comments | Usage |
| ------ | ------ |------|------|
| frozen onnx | **model**(str): path to frozen onnx <br> **framework_specific_info**(dict): information about model and framework <br> **kwargs**(dict): other required parameters | **Save format**: <br> frozen onnx | from neural_compressor.experimental import Quantization, common <br> quantizer = Quantization(args.config) <br> quantizer.model = model <br> q_model = quantizer.fit() <br> **model is the path of model, like ./path/to/frozen.onnx** |
| onnx.onnx_ml_pb2.ModelProto | **model**(onnx.onnx_ml_pb2.ModelProto): onnx.onnx_ml_pb2.ModelProto object <br> **framework_specific_info**(dict): information about model and framework <br> **kwargs**(dict): other required parameters | **Save format**: <br> frozen onnx | from neural_compressor.experimental import Quantization, common <br> quantizer = Quantization(args.config) <br> quantizer.model = model <br> q_model = quantizer.fit() <br> **model is onnx.onnx_ml_pb2.ModelProto object** |


### MXNet

| Model format | Parameters | Comments | Usage |
| ------ | ------ |------|------|
| mxnet.gluon.HybridBlock | **model**(mxnet.gluon.HybridBlock): mxnet.gluon.HybridBlock object <br> **framework_specific_info**(dict): information about model and framework <br> **kwargs**(dict): other required parameters | **Save format**: <br> save_path.json | from neural_compressor.experimental import Quantization, common <br> quantizer = Quantization(args.config) <br> quantizer.model = model <br> q_model = quantizer.fit() <br> **model is mxnet.gluon.HybridBlock object** |
| mxnet.symbol.Symbol | **model**(tuple): tuple of symbol, arg_params, aux_params <br> **framework_specific_info**(dict): information about model and framework <br> **kwargs**(dict): other required parameters | **Save format**: <br> save_path-symbol.json and save_path-0000.params | from neural_compressor.experimental import Quantization, common <br> quantizer = Quantization(args.config) <br> quantizer.model = model <br> q_model = quantizer.fit() <br> **model is the tuple of symbol, arg_params, aux_params** |


## Model API summary

**TBD by auto link**

