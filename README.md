# PhysIR-Splat: Physically Consistent Thermal Infrared Radiative Transfer in 3D Gaussian Splatting

This repository is organized around the CVPR 2026 paper **"PhysIR-Splat: Physically Consistent Thermal Infrared Radiative Transfer in 3D Gaussian Splatting"**.

PhysIR-Splat models thermal infrared image formation on Gaussian primitives. Each primitive can carry radiative-transfer attributes such as temperature, emissivity, and environmental irradiance.

## Method Overview

The paper describes two coupled components:

- **PhysIR-Splat**: a thermal Gaussian rendering framework based on passband Planck radiance, environmental reflection, atmospheric transmittance, and monotonic radiometric response.
- **VGGT-IR**: a Transformer-based thermal infrared initializer that directly regresses camera poses, depth, and point maps from multi-view TIR input with optional RGB.

## Dataset

The training examples use TI-NSD scenes organized as:

```shell
data/
  TI-NSD/
    apples/
    basketball_court/
    ...
```

The split follows the paper setup: with `--eval`, every 8th image is used as the test set and the remaining images are used for training.

RGB+thermal scenes are also supported when organized like RGBT-Scenes:

```shell
scene_name/
  sparse/0/                 # or colmap/sparse/0/
  rgb/
    train/
    test/
  thermal/
    train/
    test/
```

## Environment

```shell
conda create -n physir python=3.7
conda activate physir

pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
pip install -r requirements.txt
```

If you already use an existing environment, keep it active and only install missing dependencies.

## Training

```shell
python train.py -s path/to/your/TI-NSD/scene_name --eval
```

RGB+thermal:

```shell
python train.py -s path/to/your/RGBT-Scenes/scene_name --eval
```

Select a data branch explicitly:

```shell
python train.py -s path/to/your/TI-NSD/scene_name --eval --data_branch ir
python train.py -s path/to/your/RGBT-Scenes/scene_name --eval --data_branch rgbt
```

`--data_branch auto` is the default and keeps automatic scene detection.

The default schedule trains for 30k iterations.

## Rendering And Evaluation

```shell
python render.py -m output/exp-name
python metrics.py -m output/exp-name
```

## Acknowledgements

We thank the authors of the following thermal 3D Gaussian Splatting works for their inspiring contributions:

```bibtex
@inproceedings{luthermalgaussian,
  title={ThermalGaussian: Thermal 3D Gaussian Splatting},
  author={Lu, Rongfeng and Chen, Hangyu and Zhu, Zunjie and Qin, Yuhang and Lu, Ming and Yan, Chenggang and others},
  booktitle={The Thirteenth International Conference on Learning Representations}
}

@inproceedings{chen2024thermal3d,
  title={Thermal3d-gs: Physics-induced 3d gaussians for thermal infrared novel-view synthesis},
  author={Chen, Qian and Shu, Shihao and Bai, Xiangzhi},
  booktitle={European Conference on Computer Vision},
  pages={253--269},
  year={2024},
  organization={Springer}
}

@inproceedings{nam2025veta,
  title={Veta-GS: View-dependent deformable 3D Gaussian Splatting for thermal infrared Novel-view Synthesis},
  author={Nam, Myeongseok and Park, Wongi and Kim, Minsol and Hur, Hyejin and Lee, Soomok},
  booktitle={2025 IEEE International Conference on Image Processing (ICIP)},
  pages={965--970},
  year={2025},
  organization={IEEE}
}
```

## Paper Reference

```bibtex
@inproceedings{gao2026physirsplat,
  title={PhysIR-Splat: Physically Consistent Thermal Infrared Radiative Transfer in 3D Gaussian Splatting},
  author={Gao, Jingyuan and Hu, Yumeng and Gao, Fei and Zhang, Mingjin},
  booktitle={CVPR},
  year={2026}
}
```
