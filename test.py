import tensorflow as tf

class Client:
    def _init_(self, gpu_fraction):
        self.gpu_fraction = gpu_fraction

    def set_gpu_memory_fraction(self):
        gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=self.gpu_fraction)
        config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)
        session = tf.compat.v1.Session(config=config)
        tf.compat.v1.keras.backend.set_session(session)

# Esempio di utilizzo
client1 = Client(gpu_fraction=0.5)  # Imposta la percentuale massima di GPU al 50%
client2 = Client(gpu_fraction=0.3)  # Imposta la percentuale massima di GPU al 30%