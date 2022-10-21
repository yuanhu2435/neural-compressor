# 1. What's Mixed-Precision

The recent growth of Deep Learning has driven the development of more complex models that require significantly more compute and memory capabilities. Several low precision numeric formats have been proposed to address the problem. Google's [bfloat16](https://cloud.google.com/tpu/docs/bfloat16) and the [FP16: IEEE](https://en.wikipedia.org/wiki/Half-precision_floating-point_format) half-precision format are two of the most widely used sixteen bit formats. [Mixed precision](https://arxiv.org/abs/1710.03740) training and inference using low precision formats have been developed to reduce compute and bandwidth requirements.

The recently launched 3rd Gen Intel® Xeon® Scalable processor (codenamed Cooper Lake), featuring Intel® Deep Learning Boost, is the first general-purpose x86 CPU to support the bfloat16 format. Specifically, three new bfloat16 instructions are added as a part of the AVX512_BF16 extension within Intel Deep Learning Boost: VCVTNE2PS2BF16, VCVTNEPS2BF16, and VDPBF16PS. The first two instructions allow converting to and from bfloat16 data type, while the last one performs a dot product of bfloat16 pairs. Further details can be found in the [hardware numerics document](https://software.intel.com/content/www/us/en/develop/download/bfloat16-hardware-numerics-definition.html) published by Intel.

Intel® Neural Compressor (INC) supports `BF16 + FP32` mixed precision conversion by MixedPrecision API across multiple framework backends.

# 2. Accuracy-aware tuning

Mixed-precision conversion may lead to accuracy drop. INC provides anaccuracy-aware tuning function for mixed-precision to reduce accuracy loss, which will fallback converted ops to FP32 automatically to get better accuracy. To enable this function, users only need to provide an evaluation function (or dataloader + metric) to execute evaluation process.

# 3. MixedPrecision Support Matrix

|Framework     |BF16         |
|--------------|:-----------:|
|TensorFlow    |&#10004;     |
|PyTorch       |&#10004;     |
|ONNX          |plan to support in the future |
|MXNet         |&#10004;     |

> During quantization, BF16 conversion will be executed automatically as well if pre-requirements are met or force enable it. Please refer to this [document](./quantization_mixed_precision.md) for its workflow.


# 4. MixedPrecision API summary

## 4.1 Yaml Syntax Introduction

Below is an example for mixed-precision yaml.

```yaml
model:
  name: resnet50_v1
  framework: tensorflow
mixed_precision:
  precisions: 'bf16'
evaluation:
  accuracy:
    dataloader:
      ...
    metric:
      ...
  ```

|Field       |Description             |
|---------------- |:-----------|
|mixed_precision | Optional. Contains mixed_precision sub-field which can be used to set target precision. If users define target precision with python API, mixed_precision filed can be removed.|
|evaluation | Optional. Contains accuracy sub-field which can be used to set metric and dataloader. If users use their own evaluation function (or dataloader and metric) by python code, evaluation filed can be removed.|


|Sub-field       |Description             |
|---------------- |:-----------|
|precisions | Optional.Target precision for conversion, INC will convert as many ops as possible to target precision. It supports str and list of str.|
|accuracy | Optional. Set built-in dataloader and metric which are used to do accuracy-aware tuning for mixed-precision. Please refer to [dataloader doc](./dataloader.md) and [metric doc](./metric.md) to learn how to use built-in dataloader and metric.|


## 4.2 API Introduction

INC provides MixedPrecision API to do mixed-precision conversion.

|Parameters       |             |
|---------------- |:-----------|
|conf_fname_or_obj| Optional. It supports yaml file path, INC config class and None|


|Attributes      |             |
|----------------|:-----------|
|precisions      |Target precision     |
|input           |User-specified input     |
|output          |Plan to support in the future |
|eval_dataloader |User-specified eval_dataloader, it needs to co-work with metric     |
|model           |Candidate model to do mixed-precision conversion|
|metric          |User-specified metric, it needs to co-work with eval_dataloader|
|eval_func       |User-specified eval_func, it will overide metric and eval_dataloader|


# 5. Examples

There are 2 pre-requirements to run BF16 mixed-precision examples:

- Hardware: CPU supports `avx512_bf16` instruction set.
- Software: intel-tensorflow >= [2.3.0](https://pypi.org/project/intel-tensorflow/2.3.0/) or torch > [1.11.0](https://download.pytorch.org/whl/torch_stable.html).

If either pre-requirement can't be met, the program would exit consequently. Otherwise, we can force enable BF16 conversion for debug usage by setting the environment variable `FORCE_BF16=1`:
```shell
FORCE_BF16=1 /path/to/executable_nc_wrapper
```
> ⚠️Without hardware or software support, the poor performance or other problems are expected for force enabling.

## 5.1 TensorFlow Simple Example



## 5.2 PyTorch simple example

