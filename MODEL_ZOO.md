# Model Zoo

Some note:

-  `Frame = input_frame x crop x clip`
  - `input_frame` means how many frames are input for model per inference
  - `crop` means spatial crops (e.g., 3 for left/right/center)
  - `clip` means temporal clips (e.g., 4 means repeted sampling four clips with different start indices)

## Kinetics-400

|       Model      |    Depth   | Dim | Kernels | Top-1 | Checkpoints | Config | Params (MB)
|:----------------:|:----------:|:---:|:-------:|:-----:|:--------:|:--------:|:--------:|
| Video-FocalNet-T |  [2,2,6,2] |  96 |  [3,5]  |  79.8 |   [SavedModel]()/[H5]()   |   [cfg](https://github.com/TalalWasim/Video-FocalNets/blob/main/configs/kinetics400/video-focalnet_tiny.yaml)   |   49.55   |
| Video-FocalNet-S | [2,2,18,2] |  96 |  [3,5]  |  81.4 |   [SavedModel]()/[H5]()   |   [cfg](https://github.com/TalalWasim/Video-FocalNets/blob/main/configs/kinetics400/video-focalnet_small.yaml)   |   88.74   |
| Video-FocalNet-B | [2,2,18,2] | 128 |  [3,5]  |  83.6 |   [SavedModel]()/[H5]()   |   [cfg](https://github.com/TalalWasim/Video-FocalNets/blob/main/configs/kinetics400/video-focalnet_base.yaml)   |   157.03   |

## Kinetics-600

|       Model      |    Depth   | Dim | Kernels | Top-1 | Checkpoints | Config | Params (MB)
|:----------------:|:----------:|:---:|:-------:|:-----:|:--------:|:--------:|:--------:|
| Video-FocalNet-B | [2,2,18,2] | 128 |  [3,5]  |  86.7 |   [SavedModel]()/[H5]()   |   [cfg](https://github.com/TalalWasim/Video-FocalNets/blob/main/configs/kinetics600/video-focalnet_base.yaml)   |   157.03   |

## Something-Something-v2

|       Model      |    Depth   | Dim | Kernels | Top-1 | Checkpoints | Config | Params (MB)
|:----------------:|:----------:|:---:|:-------:|:-----:|:--------:|:--------:|:--------:|
| Video-FocalNet-B | [2,2,18,2] | 128 |  [3,5]  |  71.1 |   [SavedModel]()/[H5]()   |   [cfg](https://github.com/TalalWasim/Video-FocalNets/blob/main/configs/ssv2/video-focalnet_base.yaml)   |   157.03   |

## Diving-48

|       Model      |    Depth   | Dim | Kernels | Top-1 | Checkpoints | Config | Params (MB)
|:----------------:|:----------:|:---:|:-------:|:-----:|:--------:|:--------:|:--------:|
| Video-FocalNet-B | [2,2,18,2] | 128 |  [3,5]  |  90.8 |   [SavedModel]()/[H5]()   |   [cfg](https://github.com/TalalWasim/Video-FocalNets/blob/main/configs/diving48/video-focalnet_base.yaml)   |   157.03   |

## ActivityNet-v1.3

|       Model      |    Depth   | Dim | Kernels | Top-1 | Checkpoints | Config | Params (MB)
|:----------------:|:----------:|:---:|:-------:|:-----:|:--------:|:--------:|:--------:|
| Video-FocalNet-B | [2,2,18,2] | 128 |  [3,5]  |  89.8 |   [SavedModel]()/[H5]()   |   [cfg](https://github.com/TalalWasim/Video-FocalNets/blob/main/configs/activitynet/video-focalnet_base.yaml)   |   157.03  |
