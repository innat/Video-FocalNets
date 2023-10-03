
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class TFPatchEmbed(keras.Model):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """
    def __init__(
        self, 
        img_size=(224, 224), 
        patch_size=4, 
        in_chans=3, 
        embed_dim=96,
        use_conv_embed=False, 
        norm_layer=None, 
        is_stem=False, 
        tubelet_size=1, 
        **kwargs
    ):
        super().__init__(**kwargs)

        patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.tubelet_size = tubelet_size

        if use_conv_embed:
            if is_stem:
                kernel_size = 7
                padding = 'valid'
                stride = 4
            else:
                kernel_size = 3
                padding = 'same'
                stride = 2
            self.proj = layers.Conv2D(
                embed_dim, 
                kernel_size, 
                strides=stride, 
                padding=padding
            )
        else:
            if tubelet_size == 1:
                self.proj = layers.Conv2D(
                    embed_dim, 
                    patch_size, 
                    strides=patch_size
                )
            else:
                self.proj = layers.Conv3D(
                    embed_dim, 
                    (tubelet_size, *patch_size), 
                    strides=(tubelet_size, *patch_size)
                )
        
        if norm_layer:
            self.norm = norm_layer(axis=-1, epsilon=1e-05)
        else:
            self.norm = None

    def call(self, x):
        
        if self.tubelet_size == 1:
            x = self.proj(x)
            input_shape = tf.shape(x)
            B,H,W,C = (
                input_shape[0],
                input_shape[1],
                input_shape[2],
                input_shape[3],
            )

            x = tf.reshape(
                x, [B, H*W, C]
            )
            
            if self.norm is not None:
                x = self.norm(x)
            
            return x, H, W

        else:
            x = self.proj(x)
            input_shape = tf.shape(x)
            B,T,H,W,C = (
                input_shape[0],
                input_shape[1],
                input_shape[2],
                input_shape[3],
                input_shape[4],
            )
            x = tf.reshape(
                x, [B*T, H*W, C]
            )

            if self.norm is not None:
                x = self.norm(x)
 
            return x, H, W