# Video-FocalNets

Keras Implementation of [**Video-FocalNets: Spatio-Temporal Focal Modulation for Video Action Recognition [ICCV 2023]**](https://arxiv.org/abs/2307.06947). The official PyTorch implementation is [here](https://github.com/TalalWasim/Video-FocalNets).

![](./assets/overall_architecture.png)


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

## Model Zoo

### Kinetics-400

|       Model      |    Depth   | Dim | Kernels | Top-1 | Download |
|:----------------:|:----------:|:---:|:-------:|:-----:|:--------:|
| Video-FocalNet-T |  [2,2,6,2] |  96 |  [3,5]  |  79.8 |   ?   |
| Video-FocalNet-S | [2,2,18,2] |  96 |  [3,5]  |  81.4 |   ?   |
| Video-FocalNet-B | [2,2,18,2] | 128 |  [3,5]  |  83.6 |   ?   |

### Kinetics-600

|       Model      |    Depth   | Dim | Kernels | Top-1 | Download |
|:----------------:|:----------:|:---:|:-------:|:-----:|:--------:|
| Video-FocalNet-B | [2,2,18,2] | 128 |  [3,5]  |  86.7 |   ?   |

### Something-Something-v2

|       Model      |    Depth   | Dim | Kernels | Top-1 | Download |
|:----------------:|:----------:|:---:|:-------:|:-----:|:--------:|
| Video-FocalNet-B | [2,2,18,2] | 128 |  [3,5]  |  71.1 |   ?   |
