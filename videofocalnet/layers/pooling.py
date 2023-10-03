
import math
from tensorflow.keras import layers

class TFAdaptiveAveragePooling1D(layers.Layer):
    def __init__(self, output_size, **kwargs):
        super().__init__(**kwargs)
        self.output_size = output_size

    def call(self, inputs):
        output_size = self.output_size
        input_dim = inputs.shape[1]
        ksize = math.ceil(input_dim / output_size)
        return layers.AveragePooling1D(
            pool_size=[ksize], strides=[ksize], padding="valid"
        )(inputs)
    