Small NN library in C# with syntax similar to Keras.

Has support for 32-bit quantized fixed-point forward flow and quantization-aware-training.

In QAT mode, forward propagation will use 32-bit integer math while backpropagation will use FP math.

This is suitable for edge-device logic hardware supported by 32bit MCU.
