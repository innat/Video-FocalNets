# Video-FocalNets

Keras Implementation of [**Video-FocalNets: Spatio-Temporal Focal Modulation for Video Action Recognition [ICCV 2023]**](https://arxiv.org/abs/2307.06947). The official PyTorch implementation is [here](https://github.com/TalalWasim/Video-FocalNets).

```python
def tf_videofocalnet_tiny(**kwargs):
    model = TFVideoFocalNet(depths=[2, 2, 6, 2], embed_dim=96, **kwargs)
    return model

num_frames = 8
model = tf_videofocalnet_tiny(num_classes=400)
y_pred = model_tf(tf.ones(shape=(1, num_frames, 224, 224, 3)))
y_pred.shape # TensorShape([1, 400])

n_parameters = model_tf.count_params()
print("%.2f" % (n_parameters / 1.0e6)) 
#  49.55
```
