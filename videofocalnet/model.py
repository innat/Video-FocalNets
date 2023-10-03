
import os
from functools import partial
import warnings

warnings.simplefilter(action="ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras import layers

from layers import TFPatchEmbed
from layers import TFAdaptiveAveragePooling1D
from blocks import TFBasicLayer

class TFVideoFocalNet(keras.Model):
    r"""Spatio Temporal Focal Modulation Networks (Video-FocalNets)

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Focal Transformer layer.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        drop_rate (float): Dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        focal_levels (list): How many focal levels at all stages. 
            Note that this excludes the finest-grain level. Default: [1, 1, 1, 1] 
        focal_windows (list): The focal window size at all stages. Default: [7, 5, 3, 1] 
        use_conv_embed (bool): Whether use convolutional embedding. 
            We noted that using convolutional embedding usually improve the performance, 
            but we do not use it by default. Default: False 
        use_layerscale (bool): Whether use layerscale proposed in CaiT. Default: False 
        layerscale_value (float): Value for layer scale. Default: 1e-4 
        use_postln (bool): Whether use layernorm after modulation (it helps stablize training of large models)
    """
    def __init__(
        self,
        img_size=224, 
        patch_size=4, 
        in_chans=3, 
        num_classes=1000,
        embed_dim=96, 
        depths=[2, 2, 6, 2], 
        mlp_ratio=4., 
        drop_rate=0., 
        drop_path_rate=0.1,
        norm_layer=partial(layers.LayerNormalization, epsilon=1e-05),
        patch_norm=True,               
        focal_levels=[2, 2, 2, 2], 
        focal_windows=[3, 3, 3, 3], 
        use_conv_embed=False, 
        use_layerscale=False, 
        layerscale_value=1e-4, 
        use_postln=False, 
        use_postln_in_modulation=False, 
        normalize_modulator=False,
        num_frames=8,
        tubelet_size=1,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        # variables
        self.num_layers = len(depths)
        embed_dim = [embed_dim * (2 ** i) for i in range(self.num_layers)]
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_features = embed_dim[-1]
        self.mlp_ratio = mlp_ratio
        self.tubelet_size = tubelet_size
        self.num_frames = num_frames//self.tubelet_size
        
        # image embedding
        self.patch_embed = TFPatchEmbed(
            img_size=(img_size, )*2, 
            patch_size=patch_size, 
            in_chans=in_chans, 
            embed_dim=embed_dim[0], 
            use_conv_embed=use_conv_embed, 
            norm_layer=norm_layer if self.patch_norm else None, 
            is_stem=True,
            tubelet_size=tubelet_size,
            name='TFPatchEmbed'
        )
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution
        self.pos_drop = layers.Dropout(drop_rate)

        # stochastic depth
        dpr = tf.linspace(0., drop_path_rate, sum(depths)).numpy().tolist()
        
        # build layers
        self.basic_layers = []
        for i_layer in range(self.num_layers):
            layer = TFBasicLayer(
                dim=embed_dim[i_layer], 
                out_dim=embed_dim[i_layer+1] if (i_layer < self.num_layers - 1) else None,  
                input_resolution=(
                    patches_resolution[0] // (2 ** i_layer),
                    patches_resolution[1] // (2 ** i_layer)
                ),
                depth=depths[i_layer],
                mlp_ratio=self.mlp_ratio,
                drop=drop_rate, 
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer, 
                downsample=TFPatchEmbed if (i_layer < self.num_layers - 1) else None,
                focal_level=focal_levels[i_layer], 
                focal_window=focal_windows[i_layer], 
                use_conv_embed=use_conv_embed,
                use_layerscale=use_layerscale, 
                layerscale_value=layerscale_value, 
                use_postln=use_postln,
                use_postln_in_modulation=use_postln_in_modulation, 
                normalize_modulator=normalize_modulator,
                num_frames=self.num_frames,
                name=f'TFBasicLayer{i_layer+1}'
            )
            self.basic_layers.append(layer)
            
        self.norm = norm_layer(axis=-1, name='norm')
        self.avgpool = TFAdaptiveAveragePooling1D(1, name='adt1d_avg_pool')
        self.head = layers.Dense(num_classes, name='head', dtype='float32') if num_classes > 0 else layers.Identity()

    def forward_features(self, x):
        x, H, W = self.patch_embed(x)
        x = self.pos_drop(x)
        
        for layer in self.basic_layers:
            x, H, W = layer(x, H, W)

        x = self.norm(x)
        x = self.avgpool(x)
        x = tf.reshape(x, [tf.shape(x)[0], -1])
        
        return x
    
    def call(self, x):
        input_shape = tf.shape(x)
        b,t,h,w,c = (
            input_shape[0],
            input_shape[1],
            input_shape[2],
            input_shape[3],
            input_shape[4]
        )
        
        if self.tubelet_size==1:
            x = tf.reshape(
                x, [-1, h, w, c]
            )
            
        # forward passing    
        x = self.forward_features(x)
        
        # Here just aggregate the corresponding frames of same video BxT, C
        x = tf.reshape(x, [b, self.num_frames, -1])
        x = tf.reduce_mean(x, axis=1)
        x = self.head(x)
        return x
    
    def build(self, input_shape):
        super().build(input_shape)
        self.build_shape = input_shape[1:]

    def build_graph(self):
        x = keras.Input(shape=self.build_shape, name='input_graph')
        return keras.Model(
            inputs=[x], outputs=self.call(x)
        )