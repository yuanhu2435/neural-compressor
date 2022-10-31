Pruning
============

## 1 What is weight pruning

Network pruning is one of popular approaches of network compression, which removes the least important parameters in the network to achieve compact architectures with minimal accuracy drop.

### 1.1 Unstructured Pruning

Unstructured pruning means finding and removing the less salient connection in the model where the nonzero patterns are irregular and could be anywhere in the matrix.

### 1.2 Structured Pruning

Structured pruning means finding parameters in groups, deleting entire blocks, filters, or channels according to some pruning criterions. \
Here is a figure showing a matrix with ```IC``` = 32 and ```OC``` = 16 dimension, and a block-wise sparsity pattern with block size 4 on ```OC``` dimension.
<a target="_blank" href="./docs/imgs/sparse_dim.png">
    <img src="../docs/imgs/sparse_dim.png" width=854 height=479 alt="Sparsity Pattern">
</a>

In general, structured sparsity has lower accuracy due to restrictive structure than unstructured sparsity; however, it can accelerate the model execution significantly with software or hardware sparsity.

## 2 Pruning support matrix
<table>
<thead>
  <tr>
    <th>Pruning Type</th>
    <th>Pruning Granularity</th>
    <th>Pruning Algorithm</th>
    <th>Framework</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td rowspan="3">Unstructured Pruning</td>
    <td rowspan="3">Element-wise</td>
    <td>Magnitude</td>
    <td>PyTorch, TensorFlow</td>
  </tr>
  <tr>
    <td>Pattern Lock</td>
    <td>PyTorch</td>
  </tr>
  <tr>
    <td>SNIP with momentum</td>
    <td>PyTorch</td>
  </tr>
  <tr>
    <td rowspan="6">Structured Pruning</td>
    <td rowspan="2">Filter/Channel-wise</td>
    <td>Gradient Sensitivity</td>
    <td>PyTorch</td>
  </tr>
  <tr>
    <td>SNIP with momentum</td>
    <td>PyTorch</td>
  </tr>
  <tr>
    <td rowspan="2">Block-wise</td>
    <td>Group Lasso</td>
    <td>PyTorch</td>
  </tr>
  <tr>
    <td>SNIP with momentum</td>
    <td>PyTorch</td>
  </tr>
  <tr>
    <td rowspan="2">Element-wise</td>
    <td>Pattern Lock</td>
    <td>PyTorch</td>
  </tr>
  <tr>
    <td>SNIP with momentum</td>
    <td>PyTorch</td>
  </tr>
</tbody>
</table>

- Magnitude

  - The algorithm prunes the weight by the lowest absolute value at each layer with given sparsity target.

- Gradient sensitivity

  - The algorithm prunes the head, intermediate layers, and hidden states in NLP model according to importance score calculated by following the paper [FastFormers](https://arxiv.org/abs/2010.13382). 

- Group Lasso

  - The algorithm uses Group lasso regularization to prune entire rows, columns or blocks of parameters that result in a smaller dense network.

- Pattern Lock

  - The algorithm locks the sparsity pattern in fine tune phase by freezing those zero values of weight tensor during weight update of training. 

- SNIP

  - The algorithm prunes the dense model at its initialization, by analyzing the weights' effect to the loss function when they are masked. Please refer to the original [paper](https://arxiv.org/abs/1810.02340) for details

- SNIP with momentum

  - The algorithm improves original SNIP algorithms and introduces weights' score maps which updates in a momentum way.\
  In the following formula, $n$ is the pruning step and $W$ and $G$ are model's weights and gradients respectively.
  $$Score_{n} = 1.0 \times Score_{n-1} + 0.9 \times |W_{n} \times G_{n}|$$

## 3 Pruning API Summary

### User facing API

Neural Compressor pruning API is defined under `neural_compressor.experimental.Pruning`, which takes a user defined yaml file as input. The user defined yaml defines training, pruning and evaluation behaviors.
[API Readme](../docs/pruning_api.md).

### Usage 1: Launch pruning with user-defined yaml

#### Launcher code

Below is the launcher code if training behavior is defined in user-defined yaml.

```
from neural_compressor.experimental import Pruning
prune = Pruning('/path/to/user/pruning/yaml')
prune.model = model
model = prune.fit()
```

#### User-defined yaml

The user-defined yaml follows below syntax, note `train` section is optional if user implements `pruning_func` and sets to `pruning_func` attribute of pruning instance.
User could refer to the yaml files in examples to know field meanings.

##### `train`

The `train` section defines the training behavior, including what training hyper-parameter would be used and which dataloader is used during training. 

##### `approach`

The `approach` section defines which pruning algorithm is used and how to apply it during training process.

- ``weight compression``: pruning target, currently only ``weight compression`` is supported. ``weight compression`` means zeroing the weight matrix. The parameters for `weight compression` is divided into global parameters and local parameters in different ``pruners``. Global parameters may contain `start_epoch`, `end_epoch`, `initial_sparsity`, `target_sparsity` and `frequency`. 

  - `start_epoch`:  on which epoch pruning begins
  - `end_epoch`: on which epoch pruning ends
  - `initial_sparsity`: initial sparsity goal, default 0.
  - `target_sparsity`: target sparsity goal
  - `frequency`: frequency to update sparsity.

- `Pruner`:

  - `prune_type`: pruning algorithm, currently ``basic_magnitude``, ``gradient_sensitivity`` and ``group_lasso``are supported.

  - `names`: weight name to be pruned. If no weight is specified, all weights of the model will be pruned.

  - `parameters`: Additional parameters is required ``gradient_sensitivity`` prune_type, which is defined in ``parameters`` field. Those parameters determined how a weight is pruned, including the pruning target and the calculation of weight's importance. It contains:

    - `target`: the pruning target for weight, will override global config `target_sparsity` if set.
    - `stride`: each stride of the pruned weight.
    - `transpose`: whether to transpose weight before prune.
    - `normalize`: whether to normalize the calculated importance.
    - `index`: the index of calculated importance.
    - `importance_inputs`: inputs of the importance calculation for weight.
    - `importance_metric`: the metric used in importance calculation, currently ``abs_gradient`` and ``weighted_gradient`` are supported.

    Take above as an example, if we assume the 'bert.encoder.layer.0.attention.output.dense.weight' is the shape of [N, 12\*64]. The target 8 and stride 64 is used to control the pruned weight shape to be [N, 8\*64]. `Transpose` set to True indicates the weight is pruned at dim 1 and should be transposed to [12\*64, N] before pruning. `importance_input` and `importance_metric` specify the actual input and metric to calculate importance matrix.

### Usage 2: Launch pruning with user-defined pruning function

#### Launcher code

In this case, the launcher code is like the following:

```python
from neural_compressor.experimental import Pruning, common
prune = Pruning(args.config)
prune.model = model
prune.train_func = pruning_func
model = prune.fit()
```

#### User-defined pruning function

User can pass the customized training/evaluation functions to `Pruning` for flexible scenarios. In this case, pruning process can be done by pre-defined hooks in Neural Compressor. User needs to put those hooks inside the training function.

Neural Compressor defines several hooks for user use:

```
on_epoch_begin(epoch) : Hook executed at each epoch beginning
on_step_begin(batch) : Hook executed at each batch beginning
on_step_end() : Hook executed at each batch end
on_epoch_end() : Hook executed at each epoch end
on_before_optimizer_step() : Hook executed after gradients calculated and before backward
```

Following section shows how to use hooks in user pass-in training function which is part of example from BERT training:

```python
def pruning_func(model):
    for epoch in range(int(args.num_train_epochs)):
        pbar = ProgressBar(n_total=len(train_dataloader), desc='Training')
        model.train()
        prune.on_epoch_begin(epoch)
        for step, batch in enumerate(train_dataloader):
            prune.on_step_begin(step)
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'labels': batch[3]}
            #inputs['token_type_ids'] = batch[2]
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            if (step + 1) % args.gradient_accumulation_steps == 0:
                prune.on_before_optimizer_step()
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
    
            prune.on_step_end()
...
```

## 4 Examples
We validate the sparsity on typical models across different domains (including CV, NLP, and Recommendation System). The below table shows the sparsity pattern, sparsity ratio, and accuracy of sparse and dense (Reference) model for each model. We also provide a simplified [BERT example](../examples/pytorch/nlp/huggingface_models/question-answering/pruning/group_lasso/eager) with only one sparse layer.

|   Model   | Sparsity Pattern | Sparsity Ratio |Dataset| Accuracy (Sparse Model) | Accuracy (Dense Model) |
|-----------|:----------------:|:--------------:|:-------------:|:-----------------------:|:-----------------------:|
| Bert Large| [***2***x1](../examples/pytorch/nlp/huggingface_models/question-answering/pruning/group_lasso/eager)          | 70%            |SQuAD| 90.70%                  | 91.34%                  |
| DLRM      | 4x***16***         | 85%            |Criteo Terabyte| 80.29%                  | 80.25%                  |
| Bert Mini | [***4***x1](../examples/pytorch/nlp/huggingface_models/text-classification/pruning/pytorch_pruner/eager)         | 90%            |MRPC| 87.22%                  | 87.52%                  |
| Bert Mini | [***4***x1](../examples/pytorch/nlp/huggingface_models/text-classification/pruning/pytorch_pruner/eager)         | 90%            |SST-2| 86.92%                  | 87.61%                  |
| Bert Mini | [***4***x1](../examples/pytorch/nlp/huggingface_models/question-answering/pruning/pytorch_pruner/eager)         | 80%            |SQuAD| 76.27%                  | 76.87%                  |
| Bert Mini | [2 in ***4***](../examples/pytorch/nlp/huggingface_models/text-classification/pruning/pytorch_pruner/eager)       | 50%            |MRPC| 86.95%                  | 87.52%                  |
| Bert Mini | [2 in ***4***](../examples/pytorch/nlp/huggingface_models/text-classification/pruning/pytorch_pruner/eager)         | 50%            |SST-2| 86.93%                  | 87.61%                  |
| Bert Mini | [2 in ***4***](../examples/pytorch/nlp/huggingface_models/question-answering/pruning/pytorch_pruner/eager)        | 50%            |SQuAD| 76.85%                  | 76.87%                  |
|ResNet50 v1.5 | [***2***x1](../examples/pytorch/image_recognition/torchvision_models/pruning/magnitude/eager)         | 78%            |Image-Net| 75.3%                  | 76.13%                  |
|SSD-ResNet34 | ***2***x1         | 75%            |Coco| 22.85%                  | 23%                  |
|ResNext101| ***2***x1         | 73%            |Image-Net| 79.14%                  | 79.37%                  |

Note: 
* ***bold*** means the sparsity dimension (```OC```).
* Bert-Mini related examples are developed based on our [Pytorch Pruner API](../neural_compressor/experimental/pytorch_pruner/). Examples of [question answering](../examples/pytorch/nlp/huggingface_models/question-answering/pruning/pytorch_pruner/eager) and [text classification](../examples/pytorch/nlp/huggingface_models/text-classification/pruning/pytorch_pruner/eager) are developed.
