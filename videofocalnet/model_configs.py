MODEL_CONFIGS = {
    # K400 (Kinetics-400)
    "FocalNetT_K400": {
        "input_shape": 224,
        "num_classes": 400,
        "num_frames": 8,
        "embed_dim": 96,
        "depths": [2, 2, 6, 2],
        "focal_levels": [2, 2, 2, 2],
        "focal_windows": [3, 3, 3, 3],
    },
    "FocalNetS_K400": {
        "input_shape": 224,
        "num_classes": 400,
        "num_frames": 8,
        "embed_dim": 96,
        "depths": [2, 2, 18, 2],
        "focal_levels": [2, 2, 2, 2],
        "focal_windows": [3, 3, 3, 3],
    },
    "FocalNetB_K400": {
        "input_shape": 224,
        "num_classes": 400,
        "num_frames": 8,
        "embed_dim": 128,
        "depths": [2, 2, 18, 2],
        "focal_levels": [2, 2, 2, 2],
        "focal_windows": [3, 3, 3, 3],
    },
    # K600 (Kinetics-600)
    "FocalNetB_K600": {
        "input_shape": 224,
        "num_classes": 600,
        "num_frames": 8,
        "embed_dim": 128,
        "depths": [2, 2, 18, 2],
        "focal_levels": [2, 2, 2, 2],
        "focal_windows": [3, 3, 3, 3],
    },
    # SSV2
    "FocalNetB_SSV2": {
        "input_shape": 224,
        "num_classes": 174,
        "num_frames": 8,
        "embed_dim": 128,
        "depths": [2, 2, 18, 2],
        "focal_levels": [2, 2, 2, 2],
        "focal_windows": [3, 3, 3, 3],
    },
    # D8 (Driving 48)
    "FocalNetB_D48": {
        "input_shape": 224,
        "num_classes": 48,
        "num_frames": 8,
        "embed_dim": 128,
        "depths": [2, 2, 18, 2],
        "focal_levels": [2, 2, 2, 2],
        "focal_windows": [3, 3, 3, 3],
    },
    # ANET (ActivityNet)
    "FocalNetB_ANET": {
        "input_shape": 224,
        "num_classes": 200,
        "num_frames": 8,
        "embed_dim": 128,
        "depths": [2, 2, 18, 2],
        "focal_levels": [2, 2, 2, 2],
        "focal_windows": [3, 3, 3, 3],
    },
}
