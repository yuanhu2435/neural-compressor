
# What's INC
Intel® Neural Compressor provide deep-learning model compression techniques like quantization, Knowledge distillation, pruning/sparsity and Neural Architecture Search (NAS). These features are already validated on Intel CPU/GPU, quantization is validated on broad hardware platforms: AMD CPU/ Arm CPU/Nvidia GPU (OnnxRuntime CUDA ExtensionProvider). Intel® Neural Compressor support different deep-learning frameworks via unified interfaces, users can define their own evaluation function to support various models. For quantization, 420+ examples are validated with a performance speedup geomean of 2.2x and up to 4.2x on Intel VNNI. Over 30 pruning and knowledge distillation samples are also available. 

Neural Coder automatically insert quantization code on a PyTorch model script with one-line API, this feature can increase the productivity. Intel® Neural Compressor provide other no-code features like GUI, users can do basic optimization, upload the models and click buttons, optimized models and performance results will be generated.


# Architecture
<a target="_blank" href="../docs/imgs/architecture.png">
  <img src="../docs/imgs/architecture.png" alt="Architecture" width=914 height=370>
</a>

Intel® Neural Compressor has unified interfaces and dispatch to different frameworks via adaptors. Each adaptor have own strategies and the strategy module contains model configs and tuning configs. Model configs define the quantization approach, if it's post-training static quantization, users need to set more parameters like calibration and others. There are several tuning strategies like basic (default) and tuning configs should choose one of them.

# Supported feature matrix
* Quantization - QAT, PTQ, Dynamic Quantization
* Mixed precision - int8, bf16
* Distillation
* NAS - basic and DyNAS
* Orchestration - One-shot and Multi-shot for combination of optimizations
* Pruning/Sparsity - unstructured and structured (n in m, nxm...)
* Neural Coder

# Advanced items
## Strategy 
* Default(basic) strategy: try to go through all tuning configs and select best one, if the accuracy loss exceeded the requirement, fallback each op to bf16 or fp32 and sort the ops with impact. Finally, fallback the ops 
* Bayesian: Constructing posterior distribution of functions (gaussian process) that best describes the function you want to optimize. As the number of observations grows, the posterior distribution improves, certain regions in parameter space are worth exploring and the algorithm will focus on them. Then choose the configuration that maximizes the expected improvement.  
* MSE:  Get the tensors for each operator of raw FP32 models and the quantized model with best tuning configuration. Calculates the MSE (Mean Squared Error) for each operator, sorts those operators according to the MSE value, and performs the op-wise fallback in this order.
* SigOpt: SigOpt Experiments can optimize any real-valued objective function and understand which metric or metrics to optimize. In Neural Compressor sigopt strategy, the metrics add accuracy as constraint and optimize for latency.
