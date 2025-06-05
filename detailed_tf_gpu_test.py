import tensorflow as tf
import os

print(f"TensorFlow Version: {tf.__version__}")
print(f"LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH')}")

# Try to load a specific CUDA library that TensorFlow uses, to get a more direct error
from ctypes.util import find_library
print(f"Looking for libcudnn.so.8: {find_library('libcudnn.so.8')}")
print(f"Looking for libcublas.so.12: {find_library('libcublas.so.12')}")
print(f"Looking for libcuda.so.1: {find_library('libcuda.so.1')}") # This is the CUDA driver API library

print("\nAttempting to list physical GPUs...")
try:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"SUCCESS: Num GPUs Available: {len(gpus)}")
        for gpu in gpus:
            print(f"  GPU Name: {gpu.name}, Details: {gpu}")
    else:
        print("FAILURE: No GPUs were detected by TensorFlow.")
except Exception as e:
    print(f"ERROR during GPU detection: {e}")

print("\nTensorFlow's view of CUDA availability:")
print(f"tf.test.is_built_with_cuda(): {tf.test.is_built_with_cuda()}")
print(f"tf.test.is_built_with_gpu_support(): {tf.test.is_built_with_gpu_support()}")
if tf.test.is_built_with_cuda():
    print(f"tf.test.gpu_device_name(): {tf.test.gpu_device_name() if len(tf.config.list_physical_devices('GPU')) > 0 else 'No GPU device name found by tf.test'}")