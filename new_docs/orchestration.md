Optimization Orchestration
============

## What is orchestration

Orchestration is the combination of multiple optimization techniques, either applied simultaneously (one-shot) or sequentially (multi-shot). Intel Neural Compressor supports arbitrary meaningful combinations of supported optimization methods under one-shot or multi-shot, such as pruning during quantization-aware training, or pruning and then post-training quantization, pruning and then distillation and then quantization.

### One-shot
Since quantization-aware training, pruning and distillation all leverage training process for optimization, we can achieve the goal of optimization through one shot training with arbitrary meaningful combinations of these methods, which often gain more benefits in terms of performance and accuracy than just one compression technique applied, and usually are as efficient as applying just one compression technique. The three possible combinations are shown below.
- Pruning during quantization-aware training
- Distillation with pattern lock pruning
- Distillation with pattern lock pruning and quantization-aware training
 
### Multi-shot
Of course, besides one-shot, we also support separate execution of each optimization process.
- Pruning and then post-training quantization
- Distillation and then post-training quantization
- Distillation, then pruning and post-training quantization

## Orchestration user facing API summary

Neural Compressor defines `Scheduler` class to automatically pipeline execute model optimization with one shot or multiple shots way. 

User instantiates model optimization components, such as quantization, pruning, distillation, separately. After that, user could append
those separate optimization objects into scheduler's pipeline, the scheduler API executes them one by one.

In following example it executes the pruning and then post-training quantization with two-shot way.

```python
from neural_compressor.experimental import Quantization, Pruning, Scheduler
prune = Pruning(prune_conf)
quantizer = Quantization(post_training_quantization_conf)
scheduler = Scheduler()
scheduler.model = model
scheduler.append(prune)
scheduler.append(quantizer)
opt_model = scheduler.fit()
```

If user wants to execute the pruning and quantization-aware training with one-shot way, the code is like below.

```python
from neural_compressor.experimental import Quantization, Pruning, Scheduler
prune = Pruning(prune_conf)
quantizer = Quantization(quantization_aware_training_conf)
scheduler = Scheduler()
scheduler.model = model
combination = scheduler.combine(prune, quantizer)
scheduler.append(combination)
opt_model = scheduler.fit()
```

## Examples

For orchestration one-shot related examples, please refer to [Prune Once For All SQuAD example](../examples/pytorch/nlp/huggingface_models/question-answering/optimization_pipeline/prune_once_for_all/fx/README.md) and [Prune Once For All GLUE example](../examples/pytorch/nlp/huggingface_models/text-classification/optimization_pipeline/prune_once_for_all/fx/README.md).

For orchestration multi-shot related examples, please refer to [Multi-shot examples](../examples/pytorch/image_recognition/torchvision_models/optimization_pipeline/).

### Publications
All the experiments from [Prune Once for ALL](https://arxiv.org/abs/2111.05754) can be reproduced using [Optimum-Intel](https://github.com/huggingface/optimum-intel) with Intel Neural Compressor.
