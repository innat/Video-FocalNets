# Video-FocalNets

![](./assets/overall_architecture.png)

[![Palestine](https://img.shields.io/badge/Free-Palestine-white?labelColor=green)](https://twitter.com/search?q=%23FreePalestine&src=typed_query)

[![arXiv](https://img.shields.io/badge/arXiv-2307.06947-darkred)](https://arxiv.org/abs/2307.06947) [![keras-2.12.](https://img.shields.io/badge/keras-2.12-darkred)]([?](https://img.shields.io/badge/keras-2.12-darkred)) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1fudyFyjxpoH4JGoPpuUA6PfEq0VChtT0?usp=sharing) [![HugginFace badge](https://img.shields.io/badge/🤗%20Hugging%20Face-Spaces-yellow.svg)](https://huggingface.co/spaces/innat/Video-FocalNet) [![HugginFace badge](https://img.shields.io/badge/🤗%20Hugging%20Face-Hub-yellow.svg)](https://github.com/innat/Video-FocalNets)


**Video-FocalNet** is an architecture for efficient video recognition that is effectively modeled on both local and global contexts. A spatio-temporal focal modulation approach is utilized, in which self-attention steps are optimized for greater efficiency through cost-effective convolution and element-wise multiplication. After extensive exploration, the parallel spatial and temporal encoding was determined to be the best design choice.

This is a unofficial `Keras` implementation of **Video-FocalNet**. The official `PyTorch` implementation is [here](https://github.com/TalalWasim/Video-FocalNets)

## News
- **[31-10-2023]:** GPU(s), TPU-VM for fine-tune training are supported, [colab](https://github.com/innat/Video-FocalNets/blob/main/notebooks/video_focalnet_video_classification.ipynb).
- **[31-10-2023]:**  Video-FocalNet checkpoints for [Driving-48](http://www.svcl.ucsd.edu/projects/resound/dataset.html) becomes available, [link](https://github.com/innat/Video-FocalNets/releases/tag/v1.0).
- **[30-10-2023]:**  Video-FocalNet checkpoints for [ActivityNet](http://activity-net.org/) becomes available, [link](https://github.com/innat/Video-FocalNets/releases/tag/v1.0).
- **[29-10-2023]:**  Video-FocalNet checkpoints for [SSV2](https://developer.qualcomm.com/software/ai-datasets/something-something) becomes available, [link](https://github.com/innat/Video-FocalNets/releases/tag/v1.0).
- **[28-10-2023]:**  Video-FocalNet checkpoints for [Kinetics-600](https://paperswithcode.com/dataset/kinetics-600) becomes available, [link](https://github.com/innat/Video-FocalNets/releases/tag/v1.0).
- **[27-10-2023]:**  Video-FocalNet checkpoints for [Kinetics-400](https://paperswithcode.com/dataset/kinetics-400-1) becomes available, [link](https://github.com/innat/Video-FocalNets/releases/tag/v1.0).
- **[27-10-2023]:**  Code of **Video-FocalNet** in Keras becomes available.


# Install

```python
git clone https://github.com/innat/Video-FocalNets.git
cd Video-FocalNets
pip install -e . 
```

# Usage

The Video-FocalNet checkpoints are available in both **SavedModel** and **H5** formats. The variants of this models are tiny, small, and base. Check this [release](https://github.com/innat/Video-FocalNets/releases/tag/v1.0) and [model zoo](MODEL_ZOO.md) page to know details of it. Following are some hightlights.

**Inference**

```python
>>> from videofocalnet import VideoFocalNetT
>>> model = VideoFocalNetT(name='FocalNetT_K400')
>>> _ = model(np.ones(shape=(1, 8, 224, 224, 3)))
>>> model.load_weights('TFVideoFocalNetT_K400_8x224.h5')
>>> container = read_video('sample.mp4')
>>> frames = frame_sampling(container, num_frames=8)
>>> y = model(frames)
>>> y.shape
TensorShape([1, 400])

>>> probabilities = tf.nn.softmax(y_pred_tf)
>>> probabilities = probabilities.numpy().squeeze(0)
>>> confidences = {
    label_map_inv[i]: float(probabilities[i]) \
    for i in np.argsort(probabilities)[::-1]
}
>>> confidences
```
A classification results on a sample from [Kinetics-400](https://www.deepmind.com/open-source/kinetics).

| Video                          | Top-5 |
|:------------------------------:|:-----|
| ![](./assets/view1.gif)        | <pre>{<br>    'playing cello': 0.8959084749221802,<br>    'playing violin': 0.023411624133586884,<br>    'playing recorder': 0.0011349919950589538,<br>    'playing piano': 0.00101949623785913,<br>    'playing clarinet': 0.0009982039919123054<br>}</pre> |


**Fine Tune**

Each video-focalnet checkpoints returns logits. We can just add a custom classifier on top of it. For example:

```python
# import pretrained model, i.e.
video_focalnet = keras.models.load_model(
    'TFVideoFocalNetB_K400_8x224', compile=False
    )
video_focalnet.trainable = False

# downstream model
model = keras.Sequential([
    video_focalnet,
    layers.Dense(
        len(class_folders), dtype='float32', activation=None
    )
])
model.compile(...)
model.fit(...)
model.predict(...)
```

**Spatio-Temporal Modulator [GradCAM]**

Here are some visual demonstration of first and last layer **Spatio-Temporal Modulator** of Video-FocalNet. More details [visual-gradcam.ipynb](https://github.com/innat/Video-FocalNets/blob/main/notebooks/visual_spatio_temporal_gradcam.ipynb).

https://github.com/innat/Video-FocalNets/assets/17668390/9ac7947e-879e-477e-9d4c-dfaf5f499806


## Model Zoo

The 3D video-focalnet checkpoints are listed in [MODEL_ZOO.md](MODEL_ZOO.md). 


# TODO
- [x] Custom fine-tuning code.
- [ ] Support `Keras V3` to support multi-framework backend.
- [ ] Publish on TF-Hub.

##  Citation

If you use this video-focalnet implementation in your research, please cite it using the metadata from our `CITATION.cff` file.

```swift
@InProceedings{Wasim_2023_ICCV,
    author    = {Wasim, Syed Talal and Khattak, Muhammad Uzair and Naseer, Muzammal and Khan, Salman and Shah, Mubarak and Khan, Fahad Shahbaz},
    title     = {Video-FocalNets: Spatio-Temporal Focal Modulation for Video Action Recognition},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    year      = {2023},
}
```
