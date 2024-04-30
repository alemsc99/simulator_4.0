import tensorflow as tf
import subprocess

def check_gpu_memory():
    # Use nvidia-smi command to get GPU memory usage
    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'], stdout=subprocess.PIPE)
    return float(result.stdout.strip())

def run_with_fraction(fraction):
    # Define GPU options
    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=fraction)
    config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)

    # Create TensorFlow session with specified configuration
    with tf.compat.v1.Session(config=config) as sess:
        # Define a simple TensorFlow computation graph
        a = tf.constant(5.0)
        b = tf.constant(3.0)
        c = tf.add(a, b)
        
        # Execute the graph and fetch the result
        result = sess.run(c)
        print("Result:", result)

# Run with fraction 0.000001
print("Memory usage with fraction 0.000001:", check_gpu_memory())
run_with_fraction(0.000001)

# Run with fraction 0.01
print("Memory usage with fraction 0.01:", check_gpu_memory())
run_with_fraction(0.01)




# Run with fraction 0.01
print("Memory usage with fraction 0.5:", check_gpu_memory())
run_with_fraction(0.5)
