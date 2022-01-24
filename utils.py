import tensorflow as tf
import math
from tensorflow.python.framework import dtypes

class XavierUniformWithGain( tf.keras.initializers.GlorotUniform ):
    
    def __init__( self, seed=None, gain=1.0 ):
        super(XavierUniformWithGain, self).__init__( seed=seed )
        self.gain = gain


    def __call__(self, shape, dtype=dtypes.float32):
        """Returns a tensor object initialized as specified by the initializer.

        Args:
        shape: Shape of the tensor.
        dtype: Optional dtype of the tensor. Only floating point types are
        supported.

        Raises:
        ValueError: If the dtype is not floating point
        """
        scale = self.scale
        gain = self.gain
        limit = gain * math.sqrt(3.0 * scale)
        return self._random_generator.random_uniform(shape, -limit, limit, dtype)