
# What's INC
Intel® Neural Compressor provide deep-learning model compression techniques like quantization, Knowledge distillation, pruning/sparsity and Neural Architecture Search (NAS). These features are already validated on Intel CPU/GPU, quantization is validated on broad hardware platforms: AMD CPU/ Arm CPU/Nvidia GPU (OnnxRuntime CUDA ExtensionProvider). Intel® Neural Compressor support different deep-learning frameworks via unified interfaces, users can define their own evaluation function to support various models. For quantization, 420+ examples are validated with a performance speedup geomean of 2.2x and up to 4.2x on Intel VNNI. Over 30 pruning and knowledge distillation samples are also available. 

Neural Coder automatically insert quantization code on a PyTorch model script with one-line API, this feature can increase the productivity. Intel® Neural Compressor provide other no-code features like GUI, users can do basic optimization, upload the models and click buttons, optimized models and performance results will be generated.


# Architecture
Intel® Neural Compressor has unified interfaces and dispatch to different frameworks via adaptors. Each adaptor have own strategies and the strategy module contains model configs and tuning configs. Model configs define the quantization approach, if it's post-training static quantization, users need to set more parameters like calibration and others. There are several tuning strategies like basic (default) and tuning configs should choose one of them.

 
