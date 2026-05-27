# Assignment 04: Simplified 3D Gaussian Splatting

席越  
BZ25001010

This repository contains my implementation of Assignment 04 for Digital Image Processing. The goal is to complete a simplified 3D Gaussian Splatting pipeline on the `chair` scene: reconstruct camera poses and sparse points with COLMAP, implement the core PyTorch Gaussian projection and rasterization functions, train the simplified model, and compare it with the official 3DGS implementation.

## Requirements

The experiments were run on the remote Windows GPU machine `gpu4070` under:

```text
F:\作业\Digital_image_processing\Digital_image_Homework\Assignment_04
```

The simplified code uses:

```bash
pip install -r requirements.txt
```

Main packages are `torch`, `numpy`, `opencv-python`, `natsort`, and `tqdm`. In my remote setup, COLMAP was run in `colmap_env`, while the PyTorch experiments were run in `gdl_env` with CUDA PyTorch 2.6.0/cu124.

For the official 3DGS comparison, I used `graphdeco-inria/gaussian-splatting` in the same course workspace. The official CUDA extensions were compiled in `gdl_env` after adding the Conda CUDA 12.4 `nvcc` and building the two extension submodules from a temporary ASCII path to avoid Windows Chinese-path compiler encoding issues.

## Running

### Task 1: COLMAP reconstruction

```bash
conda run -n colmap_env python code/mvs_with_colmap.py --data_dir data/chair
conda run -n gdl_env python code/debug_mvs_by_projecting_pts.py --data_dir data/chair
```

This creates the COLMAP sparse model and projection verification images. The text export is stored in `data/chair/sparse/0_text/`, while the full binary model and all projection frames are treated as generated artifacts.

### Task 2: Simplified 3DGS training

```bash
conda run -n gdl_env python code/train.py \
  --colmap_dir data/chair \
  --checkpoint_dir outputs/chair/checkpoints \
  --num_epochs 20 \
  --save_every 5 \
  --debug_every 5 \
  --debug_samples 4 \
  --device cuda \
  --max_points 3000 \
  --downsample_factor 8 \
  --quiet
```

The simplified implementation completes the following core parts:

- `gaussian_model.py`: normalized quaternion rotation, exponential scale, and 3D covariance construction.
- `gaussian_renderer.py`: OpenCV/COLMAP camera projection, perspective Jacobian, 3D-to-2D covariance projection, 2D Gaussian evaluation, and front-to-back alpha blending.
- `data_utils.py`: deterministic PyTorch-only point subsampling instead of PyTorch3D farthest point sampling.
- `train.py`: fixed seed, point limit, loss CSV, summaries, debug images, and lightweight checkpoint handling.

### Task 3: Official 3DGS comparison

The official repository was run with the same `chair` COLMAP scene:

```bash
python train.py \
  -s F:\作业\Digital_image_processing\Digital_image_Homework\Assignment_04\data\chair \
  -m F:\作业\Digital_image_processing\Digital_image_Homework\Assignment_04\outputs\official_3dgs\chair_1000 \
  --iterations 1000 \
  --test_iterations 1000 \
  --save_iterations 1000 \
  --checkpoint_iterations 1000 \
  --disable_viewer \
  --quiet

python render.py \
  -m F:\作业\Digital_image_processing\Digital_image_Homework\Assignment_04\outputs\official_3dgs\chair_1000 \
  --iteration 1000 \
  --skip_test \
  --quiet
```

Large official checkpoints and rendered frame directories are not meant to be submitted. I copied one representative render to `pics/official_3dgs_render_1000.png` and kept a lightweight summary in `outputs/official_3dgs_summary.txt`.

## Evaluation

I checked the implementation in three layers:

1. Static completion: no remaining placeholder tokens in the simplified code.
2. Smoke tests: covariance, projection, 2D Gaussian evaluation, and image rendering return finite tensors with expected shapes.
3. End-to-end artifacts: COLMAP sparse reconstruction, projected sparse points, simplified training loss/debug renders, and official 3DGS render output.

The main verification command is:

```bash
conda run -n gdl_env python code/smoke_test.py
```

## Results

### COLMAP and projection check

COLMAP registered all 100 input images and reconstructed 13,490 sparse 3D points. Reprojected sparse points align with the chair views, which verifies that camera intrinsics/extrinsics are being read consistently before training.

| Projection 1 | Projection 2 |
|---|---|
| ![projection 1](pics/colmap_projection_01.png) | ![projection 2](pics/colmap_projection_02.png) |

| Projection 3 | Projection 4 |
|---|---|
| ![projection 3](pics/colmap_projection_03.png) | ![projection 4](pics/colmap_projection_04.png) |

### Simplified 3DGS

The simplified model was trained with 3,000 Gaussians at `downsample_factor=8`. The loss decreased from about `0.0863` to `0.0473` over 20 epochs.

| Training loss | Debug render grid |
|---|---|
| ![loss](pics/training_loss_curve.png) | ![debug render](pics/simplified_3dgs_debug_epoch_0015.png) |

### Official 3DGS

The official implementation ran for 1,000 iterations and produced a denser point cloud with 26,263 Gaussians. The final logged loss was about `0.0256`, and a representative rendered frame is shown below.

![official render](pics/official_3dgs_render_1000.png)

### Comparison

| Item | Simplified 3DGS | Official 3DGS |
|---|---:|---:|
| Dataset | chair, 100 images | chair, 100 images |
| Initialization | COLMAP sparse points, capped to 3,000 Gaussians | COLMAP sparse points, adaptive densification |
| Training length | 20 epochs | 1,000 iterations |
| Final logged loss | 0.047307 | 0.025645 |
| Training time | 780.775 s | about 28 s progress time, about 35 s log wall span |
| Observed GPU memory | 2223.539 MB peak recorded by script | about 1838 MB observed during run |
| Final Gaussian count | 3,000 | 26,263 |
| Rendering style | pure PyTorch dense image-grid evaluation | CUDA rasterizer with tiled splatting |
| Visual quality | recognizable but blurrier and lower resolution | sharper structure, denser geometry, clearer boundaries |

The official implementation is much faster because it uses CUDA rasterization, tile-based rendering, adaptive densification/pruning, spherical harmonics color features, and optimized nearest-neighbor utilities. The simplified implementation is easier to read and useful for understanding the math, but its dense `N x H x W` evaluation is slower and less memory efficient for high resolution rendering.

## Notes

- I intentionally do not generate a PDF for this assignment; the submission artifact is this Markdown experiment report.
- Large checkpoints, full projection folders, official rendered frame folders, and official working logs are generated artifacts and are ignored by Git.
- The official build was fixed by avoiding two Windows-specific problems: system CUDA 11.4 conflicting with PyTorch cu124, and Chinese source paths breaking MSVC/NVCC object-file generation.

## References

- Kerbl et al., 3D Gaussian Splatting for Real-Time Radiance Field Rendering, SIGGRAPH 2023.
- Official implementation: <https://github.com/graphdeco-inria/gaussian-splatting>
- Course assignment skeleton: <https://github.com/YudongGuo/DIP-Teaching/tree/main/Assignments/04_3DGS>
- COLMAP: <https://colmap.github.io/>
