
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from layers import TFSpatioTemporalFocalModulation


class TFVideoFocalNetBlock(keras.Model):
    r""" Focal Modulation Network Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop (float, optional): Dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (layers.Activation, optional): Activation layer. Default: tf.nn.gelu
        norm_layer (keras.layers, optional): Normalization layer.  Default: LayerNormalization
        focal_level (int): Number of focal levels. 
        focal_window (int): Focal window size at first focal level
        use_layerscale (bool): Whether use layerscale
        layerscale_value (float): Initial layerscale value
        use_postln (bool): Whether use layernorm after modulation
    """

    def __init__(
        self, 
        dim, 
        input_resolution, 
        mlp_ratio=4., 
        drop=0.,
        drop_path=0., 
        act_layer=layers.Activation(tf.nn.gelu), 
        norm_layer=layers.LayerNormalization,
        focal_level=1, focal_window=3,
        use_layerscale=False, layerscale_value=1e-4, 
        use_postln=False, use_postln_in_modulation=False, 
        normalize_modulator=False, num_frames=8, **kwargs
    ):
        super().__init__(**kwargs)

        # variables
        self.dim = dim
        self.input_resolution = input_resolution
        self.mlp_ratio = mlp_ratio
        self.num_frames = num_frames
        self.focal_window = focal_window
        self.focal_level = focal_level
        self.use_postln = use_postln
        self.norm1 = norm_layer(axis=-1, epsilon=1e-05)
        
        # layers
        self.modulation = TFSpatioTemporalFocalModulation(
            dim, 
            proj_drop=drop, 
            focal_window=focal_window, 
            focal_level=self.focal_level, 
            use_postln_in_modulation=use_postln_in_modulation, 
            normalize_modulator=normalize_modulator,
            num_frames=self.num_frames
        )
        self.drop_path = TFDropPath(drop_path) if drop_path > 0. else layers.Identity()
        self.norm2 = norm_layer(axis=-1, epsilon=1e-05)
        self.mlp = TFMlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            drop=drop,
            act_layer=act_layer
        )
        
        if use_layerscale:
            self.gamma_1 = self.add_weight(
                name='gamma_1', 
                shape=(dim,), 
                initializer=keras.initializers.Constant(layerscale_value), 
                trainable=True
            )
            self.gamma_2 = self.add_weight(
                name='gamma_2', 
                shape=(dim,), 
                initializer=keras.initializers.Constant(layerscale_value), 
                trainable=True
            )
        else:
            self.gamma_1 = 1.0
            self.gamma_2 = 1.0
            
        self.H = None
        self.W = None

    def call(self, x):
 
        H, W = self.H, self.W
        input_shape = tf.shape(x)
        B,L,C = (
            input_shape[0],
            input_shape[1],
            input_shape[2],
        )
        shortcut = x

        # Focal Modulation
        x = x if self.use_postln else self.norm1(x)
        x = tf.reshape(x, [B, H, W, C])
        x = self.modulation(x)
        x = tf.reshape(x, [B, H * W, C])
        x = x if not self.use_postln else self.norm1(x)

        # FFN
        x = shortcut + self.drop_path(self.gamma_1 * x)
        x = x + self.drop_path(
            self.gamma_2 * (self.norm2(self.mlp(x)) if self.use_postln else self.mlp(self.norm2(x)))
        )

        return x