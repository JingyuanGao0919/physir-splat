# PhysIR-Splat: Physically Consistent Thermal Infrared Radiative Transfer in 3D Gaussian Splatting

This repository is organized around the CVPR 2026 paper **"PhysIR-Splat: Physically Consistent Thermal Infrared Radiative Transfer in 3D Gaussian Splatting"**.

In the paper, PhysIR-Splat models thermal infrared image formation on Gaussian primitives, incorporating radiative-transfer attributes such as temperature, emissivity, and environmental irradiance.

## Method Overview

The paper describes two coupled components:

- **PhysIR-Splat**: a thermal Gaussian rendering framework based on passband Planck radiance, environmental reflection, atmospheric transmittance, and monotonic radiometric response.
- **VGGT-IR**: a Transformer-based thermal infrared initializer that regresses camera poses, depth, and point maps from multi-view TIR input with optional RGB.

This release focuses on per-scene reconstruction from display-encoded images using supplied COLMAP cameras and point clouds.

### Thermal Infrared Imaging

The thermal imaging chain follows **self-emission + environmental reflection → atmospheric transmittance → visibility compositing → radiometric response**. Gaussian contributions are accumulated along each viewing ray, and a monotonic response maps the resulting thermal signal to image intensity.

The project supports two data settings:

- **TI-NSD**: infrared-only reconstruction and novel-view synthesis from grayscale thermal images.
- **RGBT**: joint RGB and thermal reconstruction from paired visible and pseudocolor thermal images.

## Dataset

Organize TI-NSD scenes as follows:

```text
data/TI-NSD/
├── apples/
│   ├── images/
│   └── sparse/0/
│       ├── cameras.bin
│       ├── images.bin
│       └── points3D.bin
├── basketball_court/
└── ...
```

Images are sorted by filename, and every 8th image is used for testing. The remaining images are used for training. This split is applied automatically.

RGB+thermal scenes use the provided train/test split:

```text
data/RGBT-Scenes/
└── scene_name/
    ├── colmap/sparse/0/       # sparse/0/ is also supported
    │   ├── cameras.bin
    │   ├── images.bin
    │   └── points3D.bin
    ├── rgb/
    │   ├── train/
    │   └── test/
    └── thermal/
        ├── train/
        └── test/
```

Keep the original image resolution and filenames. COLMAP text files are also supported. The training runner uses an existing `points3D.ply` or exports it from the supplied COLMAP point cloud.

## Environment

The code was tested on Linux with an NVIDIA RTX 3090, Python 3.7.13, PyTorch 1.12.1, and CUDA Toolkit 11.7. A CUDA-capable GPU, `nvcc`, and a compatible C++ compiler are required.

Run the following commands from the repository directory:

```bash
conda create -n physir python=3.7.13 pip -y
conda activate physir
python -m pip install 'setuptools<68' wheel
python -m pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
python -m pip install -r requirements.txt
export CUDA_HOME=/usr/local/cuda-11.7
python -m pip install --no-build-isolation ./native/rasterizer ./native/simple-knn
```

Adjust `CUDA_HOME` to your CUDA Toolkit installation. Use the bundled CUDA extensions. LPIPS downloads pretrained VGG16 parameters on first use; an offline setup requires these parameters to be cached beforehand.

## Training

Train and evaluate all TI-NSD scenes:

```bash
python run.py --dataset tinsd --data-root data/TI-NSD --output output/tinsd --gpus 0
```

RGB+thermal:

```bash
python run.py --dataset rgbt --data-root data/RGBT-Scenes --output output/rgbt --gpus 0
```

Each scene is trained from scratch for a total of **30,000 iterations**, using seed `0`. The runner handles the dataset-specific configuration, then renders and evaluates the final checkpoint automatically.

Add `--scenes apples` for a single TI-NSD scene or `--scenes Building` for a single RGBT scene. Use `--gpus 0 1` to distribute scenes across two GPUs. Logs and models are saved under `<output>/<scene>/`, with the final checkpoint in `model30k/`.

## Rendering And Evaluation

To render and evaluate a trained scene separately:

```bash
python render.py -m output/tinsd/apples/model30k --iteration 30000 --split test
python evaluate.py --model output/tinsd/apples/model30k
```

For RGBT, use the corresponding model directory, such as `output/rgbt/Building/model30k`; both thermal and RGB outputs are evaluated.

After all scenes have been evaluated, aggregate the results:

```bash
python aggregate.py --dataset tinsd --output output/tinsd
python aggregate.py --dataset rgbt --output output/rgbt
```

Evaluation reports PSNR, SSIM, and VGG LPIPS. `LPIPS_native` uses images in `[0,1]` without normalization; `LPIPS_standard` uses `normalize=True`. Results are averaged over views within each scene, then equally over scenes. Per-scene results are saved to `benchmark_metrics.json`, and dataset summaries to `scene_equal.json`.

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
@inproceedings{gao2026physir,
  title={PhysIR-Splat: Physically Consistent Thermal Infrared Radiative Transfer in 3D Gaussian Splatting},
  author={Gao, Jingyuan and Hu, Yumeng and Gao, Fei and Zhang, Mingjin},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={11818--11828},
  year={2026}
}
```

## License

Inria-derived code and CUDA extensions are subject to [LICENSE.md](LICENSE.md). The original repository's MIT license is retained in [LICENSE-MIT](LICENSE-MIT), and GLM's license is included in [copying.txt](native/rasterizer/third_party/glm/copying.txt).
