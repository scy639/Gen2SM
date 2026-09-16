# Gen2SM
Generalizable Single-view Object Pose Estimation by Two-side Generating and Matching [WACV 2025 **Oral**]

[ArXiv](https://arxiv.org/abs/2411.15860)


( The follow-up work on [Extreme-Two-View-Geometry-From-Object-Poses-with-Diffusion-Models](https://github.com/scy639/Extreme-Two-View-Geometry-From-Object-Poses-with-Diffusion-Models)  )

![poster](media/poster.jpg)


### Abstract

**Key word**: Sparse view object/camera pose estimation

> In this paper, we present a novel generalizable object pose estimation method to determine the object pose using only one RGB image. Unlike traditional approaches that rely on instance-level object pose estimation and necessitate extensive training data, our method offers generalization to unseen objects without extensive training, operates with a single reference image of the object, and eliminates the need for 3D object models or multiple views of the object. These characteristics are achieved by utilizing a diffusion model to generate novel-view images and conducting a two-sided matching on these generated images. Quantitative experiments demonstrate the superiority of our method over existing pose estimation techniques across both synthetic and real-world datasets. Remarkably, our approach maintains strong performance even in scenarios with significant viewpoint changes, highlighting its robustness and versatility in challenging conditions.

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
@misc{sun2024extreme,
      title={Extreme Two-View Geometry From Object Poses with Diffusion Models}, 
      author={Yujing Sun and Caiyi Sun and Yuan Liu and Yuexin Ma and Siu Ming Yiu},
      year={2024},
      eprint={2402.02800},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

