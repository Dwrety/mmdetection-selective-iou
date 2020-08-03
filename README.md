
# Solving Missing-Annotation Object Detection with Background Recalibration Loss

Official Implementation of [ArXiv](https://arxiv.org/abs/2002.05274).
The master branch works with **PyTorch 1.1** or higher.

- C++ code and CUDA code is available in `mmdet/ops/mirror_sigmoid_focal_loss/`
- Example settings please see `configs/BRL/brl_retinanet_r50_fpn_coco_50p.py`

## Cite

```
@article{Zhang2020SolvingMO,
  title={Solving Missing-Annotation Object Detection with Background Recalibration Loss},
  author={Han Zhang and Fangyi Chen and Zhiqiang Shen and Qiqi Hao and Chenchen Zhu and Marios Savvides},
  journal={ICASSP 2020 - 2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year={2020},
  pages={1888-1892}
}
```