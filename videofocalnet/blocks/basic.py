
from functools import partial

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import TFVideoFocalNetBlock

class TFBasicLayer(keras.Model):
    """ A basic Focal Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (keras.layers, optional): Normalization layer. Default: layers.LayerNormalization
        downsample (keras.layers | None, optional): Downsample layer at the end of the layer. Default: None
        focal_level (int): Number of focal levels
        focal_window (int): Focal window size at first focal level
        use_layerscale (bool): Whether use layerscale
        layerscale_value (float): Initial layerscale value
        use_postln (bool): Whether use layernorm after modulation
    """
    def __init__(
        self, 
        dim, 
        out_dim, 
        input_resolution, 
        depth,
        mlp_ratio=4., 
        drop=0., 
        drop_path=0., 
        norm_layer=partial(layers.LayerNormalization, epsilon=1e-05),
        downsample=None,
        focal_level=1, 
        focal_window=1,
        use_conv_embed=False,
        use_layerscale=False, 
        layerscale_value=1e-4,
        use_postln=False,
        use_postln_in_modulation=False,
        normalize_modulator=False,
        num_frames=8, 
        **kwargs
    ):
        super().__init__(**kwargs)

        # variables
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.num_frames = num_frames

        # blocks
        self.blocks = [
            TFVideoFocalNetBlock(
                dim=dim,
                input_resolution=input_resolution,
                mlp_ratio=mlp_ratio,
                drop=drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                focal_level=focal_level,
                focal_window=focal_window,
                use_layerscale=use_layerscale,
                layerscale_value=layerscale_value,
                use_postln=use_postln,
                use_postln_in_modulation=use_postln_in_modulation,
                normalize_modulator=normalize_modulator,
                num_frames=self.num_frames
            ) for i in range(depth)
        ]

        if downsample:
            self.downsample = downsample(
                img_size=input_resolution,
                patch_size=2,
                in_chans=dim,
                embed_dim=out_dim,
                use_conv_embed=use_conv_embed,
                norm_layer=norm_layer,
                is_stem=False
            )
        else:
            self.downsample = None

    def call(self, x, H, W):
        
        for i, blk in enumerate(self.blocks):
            blk.H, blk.W = H, W
            x = blk(x)

        if self.downsample is not None:
            x = tf.reshape(x, [tf.shape(x)[0], H, W, -1])
            x, Ho, Wo = self.downsample(x)
        else:
            Ho, Wo = H, W

        return x, Ho, Wo