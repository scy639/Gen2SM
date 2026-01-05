# Gen2SM
Generalizable Single-view Object Pose Estimation by Two-side Generating and Matching [WACV 2025 **Oral**]

[ArXiv](https://arxiv.org/abs/2411.15860)


( The follow-up work on [Extreme-Two-View-Geometry-From-Object-Poses-with-Diffusion-Models](https://github.com/scy639/Extreme-Two-View-Geometry-From-Object-Poses-with-Diffusion-Models)  )

![poster](media/poster.jpg)

## Setup

Please refer to [Extreme-Two-View-Geometry-From-Object-Poses-with-Diffusion-Models](https://github.com/scy639/Extreme-Two-View-Geometry-From-Object-Poses-with-Diffusion-Models)


## Infer
#### To eval on the two testsets adopted in [E2VG](https://github.com/scy639/Extreme-Two-View-Geometry-From-Object-Poses-with-Diffusion-Models):
`python eval_naviTestset.py` and `python eval_gsoTestset.py`

#### To eval on your custom testset:
1. Refer to `Dataset/gso.py` or `Dataset/navi.py` to create a new file implementing `CustomDatabase` and `CustomDataset`
2. Run `python eval_custom.py`. You may modify relevant configurations in eval_custom.py if needed.

#### Limitation
The current code version assumes that the input images:
- do not exhibit in-plane object rotation
- are captured from viewpoints on the upper hemisphere of the object (i.e., the camera is positioned above the object)


## Citation
```
@InProceedings{sun2024generalizable,
    title     = {Generalizable Single-View Object Pose Estimation by Two-Side Generating and Matching},
    author    = {Sun, Yujing and Sun, Caiyi and Liu, Yuan and Ma, Yuexin and Yiu, Siu Ming},
    booktitle = {Proceedings of the Winter Conference on Applications of Computer Vision (WACV)},
    month     = {February},
    year      = {2025},
    pages     = {545-556}
}
```

