
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class TFSpatioTemporalFocalModulation(keras.Model):
    def __init__(
        self, 
        dim, 
        focal_window, 
        focal_level, 
        focal_factor=2, 
        bias=True, 
        proj_drop=0.,
        use_postln_in_modulation=False, 
        normalize_modulator=False, 
        num_frames=8, **kwargs
    ):
        super().__init__(**kwargs)

        # variables
        self.dim = dim
        self.focal_window = focal_window
        self.focal_level = focal_level
        self.focal_factor = focal_factor
        self.use_postln_in_modulation = use_postln_in_modulation
        self.normalize_modulator = normalize_modulator
        self.num_frames = num_frames

        # placeholders
        self.kernel_sizes = []
        self.focal_layers = []
        self.focal_layers_temporal = []
        
        # layers
        self.f = layers.Dense(2*dim + (self.focal_level+1), use_bias=bias)
        self.h = layers.Conv2D(dim, kernel_size=1, strides=1, use_bias=bias)
        self.act = layers.Activation(tf.nn.gelu)
        self.proj = layers.Dense(dim)
        self.proj_drop = layers.Dropout(proj_drop)
        self.f_temporal = layers.Dense(dim + (self.focal_level+1), use_bias=bias)
        self.h_temporal = layers.Conv1D(dim, kernel_size=1, strides=1, use_bias=bias)
        
        for k in range(self.focal_level):
            kernel_size = self.focal_factor*k + self.focal_window
            self.focal_layers.append(
                keras.Sequential([
                    layers.Conv2D(
                        dim, 
                        kernel_size, 
                        strides=1,
                        padding='same', 
                        use_bias=False,
                        groups=dim, 
                        activation=tf.nn.gelu
                    )
                ])
            )
            self.kernel_sizes.append(kernel_size)
        
        for k in range(self.focal_level):
            kernel_size = self.focal_factor*k + self.focal_window
            self.focal_layers_temporal.append(
                keras.Sequential([
                    layers.Conv1D(
                        dim, 
                        kernel_size, 
                        strides=1,
                        padding='same', 
                        use_bias=False,
                        activation=tf.nn.gelu
                    )
                ])
            )

        if self.use_postln_in_modulation:
            self.ln = layers.LayerNormalization(epsilon=1e-05)

    def call(self, x):
        input_shape = tf.shape(x)
        batch_depth_size, height, width, channel = (
            input_shape[0],
            input_shape[1],
            input_shape[2],
            input_shape[3]
        )
        
        # pre linear projection temporal
        depth = self.num_frames
        batch_size = batch_depth_size // depth
        x_temporal = tf.reshape(x, [batch_size, depth, height, width, channel])
        x_temporal = tf.transpose(x_temporal, [0, 2, 3, 1, 4])
        x_temporal = tf.reshape(x_temporal, [batch_size * height * width, depth, channel])
        x_temporal = self.f_temporal(x_temporal)
        ctx_temporal, gates_temporal = tf.split(
            x_temporal, [channel, self.focal_level+1], axis=-1
        )
        gates_temporal = tf.transpose(gates_temporal, [0, 2, 1])

        # context aggregration temporal
        ctx_all_temporal = 0 
        for l in range(self.focal_level):
            ctx_temporal = self.focal_layers_temporal[l](ctx_temporal)
            ctx_all_temporal += tf.transpose(
                ctx_temporal, [0, 2, 1]
            ) * gates_temporal[:, l:l+1]

        ctx_temporal = tf.transpose(ctx_temporal, [0, 2, 1])
        ctx_global_temporal = self.act(tf.reduce_mean(ctx_temporal, axis=2, keepdims=True))
        ctx_all_temporal += ctx_global_temporal * gates_temporal[:, self.focal_level:]

        # pre linear projection spatial
        x = self.f(x)
        q, ctx, gates = tf.split(x, [channel, channel, self.focal_level+1], axis=-1)
        gates = tf.transpose(gates, [0, 3, 1, 2])

        # context aggregation spatial
        ctx_all = 0
        for l in range(self.focal_level):
            ctx = self.focal_layers[l](ctx)
            ctx_all += tf.transpose(ctx, [0, 3, 1, 2]) * gates[:, l:l+1]
            
        ctx = tf.transpose(ctx, [0, 3, 1, 2])
        ctx_global = self.act(
            tf.reduce_mean(
                tf.reduce_mean(ctx, axis=2, keepdims=True), axis=3, keepdims=True
            )
        )
        ctx_all += ctx_global * gates[:, self.focal_level:]

        # normalize context
        if self.normalize_modulator:
            ctx_all_temporal /= (self.focal_level+1)
            ctx_all /= (self.focal_level+1)

        # focal modulation
        ctx_all = tf.transpose(ctx_all, [0, 2, 3, 1])
        ctx_all_temporal = tf.transpose(ctx_all_temporal, [0, 2, 1])
        modulator_temporal = self.h_temporal(ctx_all_temporal)
        modulator_temporal = tf.reshape(modulator_temporal, [batch_size, height, width, depth, channel])
        modulator_temporal = tf.transpose(modulator_temporal, [0, 3, 1, 2, 4])
        modulator_temporal = tf.reshape(modulator_temporal, [batch_size*depth, height, width, channel])
        modulator = self.h(ctx_all)
        
        x_out = q * modulator * modulator_temporal
        if self.use_postln_in_modulation:
            x_out = self.ln(x_out)
        
        # post linear projection
        x_out = self.proj(x_out)
        x_out = self.proj_drop(x_out)
    
        return x_out