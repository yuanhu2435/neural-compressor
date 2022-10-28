INC's User YAML Configuration Files
=====

## Introducing User YAML Files

Intel® Neural Compressor (INC) uses YAML files for quick 
and user-friendly configurations. There are two types of YAML files - 
user YAML files and framework YAML files, which are used in 
running user cases and setting up framework capabilities, respectively.

First, let's take a look at a user YAML file, It defines the model, tuning
strategies, tuning calibrations and evaluations, and performance benchmarking
of the passing model.

## User YAML Syntax


A complete user YAML file is organized logically into several sections: 

* ***model***: The model specifications define a user model's name and framework
   (INC supports TensorFlow, TensorFlow_ITEX, PyTorch, PyTorch_IPEX,  ONNX Runtime, MXNet and more is yet to come). 

```yaml
model:                                               # mandatory. used to specify model specific information.
  name: mobilenet_v1 
  framework: tensorflow                              # mandatory. supported values are tensorflow, pytorch, pytorch_ipex, onnxrt_integer, onnxrt_qlinear or mxnet; allow new framework backend extension.
```
* ***quantization***: The quantization specifications define quantization's tuning space and related calibrations. To calibrate, users can 
specify *sampling_size* (optional) and use the subsection *dataloader* to specify
the dataset location using *root* and transformation using *transform*. To 
implement mode-wise constraints, users can use the subsection *model_wise* to specify 
*activation* and *weight*, etc. 
```yaml
quantization:                                        # optional. tuning constraints on model-wise for advance user to reduce tuning space.
  calibration:
    sampling_size: 20                                # optional. default value is 100. used to set how many samples should be used in calibration.
    dataloader:
      dataset:
        ImageRecord:
          root: /path/to/imagenet/                   # NOTE: modify to calibration dataset location if needed
      transform:
        BilinearImagenet: 
          height: 224
          width: 224
  model_wise:                                        # optional. tuning constraints on model-wise for advance user to reduc
    activation:
      algorithm: minmax
    weight:
      granularity: per_channel
```

* ***pruning***: The pruning specifications define pruning's tuning space. To define the training behavior, uses can 
use the subsection *train* to specify the training hyper-parameters and the training dataloader. 
To define the pruning approach, users can use the subsection *approach* to specify 
pruning target, choose the type of pruning algorithm, and the way to apply it 
during training process. 

```yaml
pruning:
  train:
    dataloader:
      batch_size: 128
      dataset:
        ImageRecord:
          root: /path/to/training/dataset             # NOTE: modify to the ImageNet training set location
      transform:
        BilinearImagenet: 
          height: 299
          width: 299
    postprocess:
      transform:
        LabelShift: 1
    epoch: 40
    optimizer:
      Adam:
        learning_rate: 1e-06
        beta_1: 0.9
        beta_2: 0.999
        epsilon: 1e-07
        amsgrad: False
    criterion:
      SparseCategoricalCrossentropy:
        reduction: sum_over_batch_size
        from_logits: False
  approach:
    weight_compression:
      initial_sparsity: 0.0
      target_sparsity: 0.54
      start_epoch: 0
      end_epoch: 19
      pruners:
        - !Pruner
            start_epoch: 0
            end_epoch: 19
            prune_type: basic_magnitude
```
* ***distillation***: The distillation specifications define distillation's tuning
space. Similar to pruning, to define the training behavior, users can use the 
subsection *train* to specify the training hyper-parameters and the training 
dataloader and it is optional if users implement *tran_func* and set the attribute
of distillation instance to *train_func*. For criterion, INC provides a built-in 
knowledge distillation loss class to calculate distillation loss.
```yaml
distillation:
  train:
    start_epoch: 0
    end_epoch: 90
    iteration: 1000
    frequency: 1
    dataloader:
      batch_size: 64
      dataset:
        ImageFolder:
          root: /path/to/dataset
      transform:
        Resize:
          size: 224
          interpolation: nearest
        KerasRescale:
          rescale: [127.5, 1]
    optimizer:
      SGD:
        learning_rate: 0.001  
        momentum: 0.1
        nesterov: True
        weight_decay: 0.001
    criterion:
      KnowledgeDistillationLoss:
        temperature: 1.0
        loss_types: ['CE', 'CE']
        loss_weights: [0.5, 0.5]
```
* ***evaluation***: The evaluation specifications define the tuning accuracy metrics and optional
  performance benchmarking configurations. The built-in accuracy metrics are topk,
  map, and f1. And users can register new metrics when the need arises. To benchmark
  performance, users can choose *cores_per_instance*, *num_of_instance*, etc.
```yaml
evaluation:                                          # optional. required if user doesn't provide eval_func in neural_compressor.Quantization.
  accuracy:                                          # optional. required if user doesn't provide eval_func in neural_compressor.Quantization.
    metric:
      topk: 1                                        # built-in metrics are topk, map, f1, allow user to register new metric.
    dataloader:
      batch_size: 32 
      last_batch: discard 
      dataset:
        ImageRecord:
          root: /path/to/imagenet/                   # NOTE: modify to evaluation dataset location if needed
      transform:
        BilinearImagenet: 
          height: 224
          width: 224
```
* ***tuning***: The tuning specifications define overall tuning targets. Users can
use *accuracy_criterion* to specify the target of accuracy loss percentage and use
*exit_policy* to specify the tuning timeout in seconds (0 is no limit). The random
seed can be specified using *randome_seed*. 

```yaml
tuning:
  accuracy_criterion:
    relative: 0.01                                  # the tuning target of accuracy loss percentage: 1%
  exit_policy:
    timeout: 0                                      # tuning timeout (seconds)
  random_seed: 9527                                 # random seed
```



## Pythonic Style Access
To meet variety of needs arising from various circumstances, INC **NOW** provides
pythonic style access - Pythonic API - for same purpose of either user or framework configurations. 
### Pythonic API Introduction

The Pythonic API for Configuration enables users to specify configurations
directly in their python codes without referring to 
a separate YAML file. While we support both simultaneously, 
the Pythonic API for Configurations has several advantages over YAML files, 
which one can tell from usages in the context below. Hence, we **recommend** 
users to use the Pythonic API for Configurations moving forward. 

Now, let's go through the Pythonic API for Configurations in the order of
sections similar as in YAML files. 

### Quantization

To specify ***quantization*** configurations, users can use the following 
Pythonic API step by step. 

* First, load the ***config*** module
```python
from neural_compressor import config
```
* Next, specify *quantization*'s inputs, outputs, tuning space, calibration, 
model-wise constraints and targets by assigning values the corresponding attributes below:
```python
config.quantization.inputs = ['image']
config.quantization.outputs = ['out']
config.quantization.backend = 'onnxrt_integerops'
config.quantization.approach = 'post_training_dynamic_quant'
config.quantization.device = 'gpu'
config.quantization.op_type_list = {'Conv': {'weight': {'dtype': ['fp32']}, 'activation': {'dtype': ['fp32']}}}
config.quantization.strategy = 'mse'
config.quantization.objective = 'accuracy'
config.quantization.timeout = 100
config.quantization.accuracy_criterion.relative = 0.5
config.quantization.reduce_range = False
config.quantization.use_bf16 = False
config.onnxruntime.precisions = ['fp32']
q = Quantization(config)
q.model = build_matmul_model()
q_model = q()
```

### Mixed-Precision
To specify ***mixed-precision*** configurations, user can assign values to the corresponding
attributes.
```python
config.quantization.device = 'cpu'
config.quantization.backend = 'pytorch'
config.quantization.approach = 'post_training_dynamic_quant'
config.quantization.use_bf16 = False
q = Quantization(config)
q.model = torch_model()
os.environ['FORCE_BF16'] = '1'
q_model = q()
del os.environ['FORCE_BF16']
```
### Distillation
To specify ***distillation*** configurations, users can assign values to 
the corresponding attributes.
```python
config.quantization.backend = 'pytorch'
distiller = Distillation(config)
model = ConvNet(16, 32)
origin_weight = copy.deepcopy(model.out.weight)
distiller.model = model
# Customized train, evaluation
distiller.teacher_model = ConvNet(16, 32)
distiller.train_func = train_func
distiller.eval_func = eval_func
model = distiller()
```
### Pruning
To specify ***pruning*** configurations, users can assign values to the corresponding attributes. 
```python
config.quantization.backend = 'pytorch'
prune = Pruning(config)
model = ConvNet(16, 32)
origin_weight = copy.deepcopy(model.out.weight)
prune.model = model
# Customized train, evaluation
prune.train_func = train_func
prune.eval_func = eval_func
model = prune()
```
### Orchestration

### Benchmark
To specify ***benchmark*** configurations, users can assign values to the
corresponding attributes.
```python
config.benchmark.cores_per_instance = 10
```
