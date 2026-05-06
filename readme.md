# FoundationPose: Unified 6D Pose Estimation and Tracking of Novel Objects
[[Paper]](https://arxiv.org/abs/2312.08344) [[Website]](https://nvlabs.github.io/FoundationPose/)

This is the official implementation of our paper to be appeared in CVPR 2024 (Highlight)

Contributors: Bowen Wen, Wei Yang, Jan Kautz, Stan Birchfield

We present FoundationPose, a unified foundation model for 6D object pose estimation and tracking, supporting both model-based and model-free setups. Our approach can be instantly applied at test-time to a novel object without fine-tuning, as long as its CAD model is given, or a small number of reference images are captured. We bridge the gap between these two setups with a neural implicit representation that allows for effective novel view synthesis, keeping the downstream pose estimation modules invariant under the same unified framework. Strong generalizability is achieved via large-scale synthetic training, aided by a large language model (LLM), a novel transformer-based architecture, and contrastive learning formulation. Extensive evaluation on multiple public datasets involving challenging scenarios and objects indicate our unified approach outperforms existing methods specialized for each task by a large margin. In addition, it even achieves comparable results to instance-level methods despite the reduced assumptions.


<img src="assets/intro.jpg" width="70%">

**🤖 For ROS version, please check [Isaac ROS Pose Estimation](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_pose_estimation), which enjoys TRT fast inference and C++ speed up.**

\
**🥇 No. 1 on the world-wide [BOP leaderboard](https://bop.felk.cvut.cz/leaderboards/pose-estimation-unseen-bop23/core-datasets/) (as of 2024/03) for model-based novel object pose estimation.**
<img src="assets/bop.jpg" width="80%">

## Demos

Robotic Applications:

https://github.com/NVlabs/FoundationPose/assets/23078192/aa341004-5a15-4293-b3da-000471fd74ed


AR Applications:

https://github.com/NVlabs/FoundationPose/assets/23078192/80e96855-a73c-4bee-bcef-7cba92df55ca


Results on YCB-Video dataset:

https://github.com/NVlabs/FoundationPose/assets/23078192/9b5bedde-755b-44ed-a973-45ec85a10bbe



# Bibtex
```bibtex
@InProceedings{foundationposewen2024,
author        = {Bowen Wen, Wei Yang, Jan Kautz, Stan Birchfield},
title         = {{FoundationPose}: Unified 6D Pose Estimation and Tracking of Novel Objects},
booktitle     = {CVPR},
year          = {2024},
}
```

If you find the model-free setup useful, please also consider cite:

```bibtex
@InProceedings{bundlesdfwen2023,
author        = {Bowen Wen and Jonathan Tremblay and Valts Blukis and Stephen Tyree and Thomas M\"{u}ller and Alex Evans and Dieter Fox and Jan Kautz and Stan Birchfield},
title         = {{BundleSDF}: {N}eural 6-{DoF} Tracking and {3D} Reconstruction of Unknown Objects},
booktitle     = {CVPR},
year          = {2023},
}
```

# Data prepare


1) Download all network weights from [here](https://drive.google.com/drive/folders/1DFezOAD0oD1BblsXVxqDsl8fj0qzB82i?usp=sharing) and put them under the folder `weights/`. For the refiner, you will need `2023-10-28-18-33-37`. For scorer, you will need `2024-01-11-20-02-45`.

1) [Download demo data](https://drive.google.com/drive/folders/1pRyFmxYXmAnpku7nGRioZaKrVJtIsroP?usp=sharing) and extract them under the folder `demo_data/`

1) [Optional] Download our large-scale training data: ["FoundationPose Dataset"](https://drive.google.com/drive/folders/1s4pB6p4ApfWMiMjmTXOFco8dHbNXikp-?usp=sharing)

1) [Optional] Download our preprocessed reference views [here](https://drive.google.com/drive/folders/1PXXCOJqHXwQTbwPwPbGDN9_vLVe0XpFS?usp=sharing) in order to run model-free few-shot version.

# Env setup option 1: docker (recommended)

**Requirements:** NVIDIA driver >= 525, CUDA 12.x, Docker with NVIDIA Container Toolkit.

### 1. Install Docker and NVIDIA Container Toolkit

```bash
# Docker
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER && newgrp docker

# NVIDIA Container Toolkit
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### 2. Build the image

The provided `docker/dockerfile` targets **CUDA 12.8 + Ubuntu 22.04** with Python 3.11 and PyTorch 2.5.1.

```bash
cd docker/
docker build -t foundationpose:latest -f dockerfile .
```

This takes ~30–60 minutes on first build (pytorch3d and nvdiffrast are compiled from source).

### 3. Launch the container

```bash
bash docker/run_container.sh
```

The script mounts your home directory and sets up X11 forwarding for visualisation. The container is named `foundationpose` and will replace any existing container with that name.

### 4. Build C++/CUDA extensions (first launch only)

Run this **inside the container**:

```bash
bash build_all.sh
```

### 5. Re-enter the container later

```bash
docker exec -it foundationpose bash
```

### Verify GPU access

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```


# Env setup option 2: conda (experimental)

- Setup conda environment

```bash
# create conda environment
conda create -n foundationpose python=3.9

# activate conda environment
conda activate foundationpose

# Install Eigen3 3.4.0 under conda environment
conda install conda-forge::eigen=3.4.0
export CMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH:/eigen/path/under/conda"

# install dependencies
python -m pip install -r requirements.txt

# Install NVDiffRast
python -m pip install --quiet --no-cache-dir git+https://github.com/NVlabs/nvdiffrast.git

# Kaolin (Optional, needed if running model-free setup)
python -m pip install --quiet --no-cache-dir kaolin==0.15.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.0.0_cu118.html

# PyTorch3D
python -m pip install --quiet --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu118_pyt200/download.html

# Build extensions
CMAKE_PREFIX_PATH=$CONDA_PREFIX/lib/python3.9/site-packages/pybind11/share/cmake/pybind11 bash build_all_conda.sh
```


# Preparing your own data

Scene directories must follow this layout:

```
my_scene/
├── cam_K.txt          # 3x3 camera intrinsics, space-separated
├── mesh/
│   └── object.obj     # CAD model of the object
├── rgb/
│   ├── 0000.png
│   ├── 0001.png
│   └── ...
└── depth/
    ├── 0000.png       # 16-bit PNG, depth in millimetres
    ├── 0001.png
    └── ...
```

RGB and depth filenames must match exactly. Depth values are divided by 1000 to convert to metres.

### Filling missing depth frames

If your depth stream dropped frames, use `fill_depth.py` to forward-fill gaps with the last known depth image.

**Dry run** (shows what would be copied, no files written):
```bash
python fill_depth.py --dry-run demo_data/my_scene
```

**Fill in place** (supports multiple scene directories at once):
```bash
python fill_depth.py demo_data/my_scene1 demo_data/my_scene2
```

# Run model-based demo

Pose estimation runs on the first frame, then automatically switches to tracking for the rest of the sequence. Visualisations are saved to `debug/`.

**Default scene (mustard bottle):**
```bash
python run_demo.py
```

**Custom scene:**
```bash
python run_demo.py \
  --mesh_file demo_data/my_scene/mesh/object.obj \
  --test_scene_dir demo_data/my_scene
```

Key arguments:
- `--est_refine_iter` — number of pose estimation refinement iterations (default 5)
- `--track_refine_iter` — number of tracking refinement iterations per frame (default 2)
- `--debug` — set to 1 to save per-frame visualisations, 0 to disable
- `--debug_dir` — output directory for visualisations (default `debug/`)


<img src="assets/demo.jpg" width="50%">


Feel free to try on other objects (**no need to retrain**) such as driller, by changing the paths in argparse.

<img src="assets/demo_driller.jpg" width="50%">


# Run on public datasets (LINEMOD, YCB-Video)

For this you first need to download LINEMOD dataset and YCB-Video dataset.

To run model-based version on these two datasets respectively, set the paths based on where you download. The results will be saved to `debug` folder
```
python run_linemod.py --linemod_dir /mnt/9a72c439-d0a7-45e8-8d20-d7a235d02763/DATASET/LINEMOD --use_reconstructed_mesh 0

python run_ycb_video.py --ycbv_dir /mnt/9a72c439-d0a7-45e8-8d20-d7a235d02763/DATASET/YCB_Video --use_reconstructed_mesh 0
```

To run model-free few-shot version. You first need to train Neural Object Field. `ref_view_dir` is based on where you download in the above "Data prepare" section. Set the `dataset` flag to your interested dataset.
```
python bundlesdf/run_nerf.py --ref_view_dir /mnt/9a72c439-d0a7-45e8-8d20-d7a235d02763/DATASET/YCB_Video/bowen_addon/ref_views_16 --dataset ycbv
```

Then run the similar command as the model-based version with some small modifications. Here we are using YCB-Video as example:
```
python run_ycb_video.py --ycbv_dir /mnt/9a72c439-d0a7-45e8-8d20-d7a235d02763/DATASET/YCB_Video --use_reconstructed_mesh 1 --ref_view_dir /mnt/9a72c439-d0a7-45e8-8d20-d7a235d02763/DATASET/YCB_Video/bowen_addon/ref_views_16
```

# Troubleshooting

- **Modern GPUs (RTX 40-series, CUDA 12.x):** use the `docker/dockerfile` in this repo — it targets CUDA 12.8 + Ubuntu 22.04 and handles the updated package names and PyTorch wheel URLs. The upstream image (`wenbowen123/foundationpose`) was built for CUDA 11.3 and will not work.

- **`TORCH_CUDA_ARCH_LIST` error during build:** Docker build has no GPU access, so the arch list must be set explicitly. The Dockerfile sets `ENV TORCH_CUDA_ARCH_LIST="8.9"` for Ada Lovelace (RTX 40-series). Change this value if you are on a different GPU generation (e.g. `8.6` for RTX 30-series, `7.5` for RTX 20-series).

- **For setting up on Windows**, refer to [this](https://github.com/NVlabs/FoundationPose/issues/148).

- **Unreasonable pose results:** check [this issue](https://github.com/NVlabs/FoundationPose/issues/44#issuecomment-2048141043) and [this manual](https://github.com/030422Lee/FoundationPose_manual).

# Training data download
Our training data include scenes using 3D assets from GSO and Objaverse, rendered with high quality photo-realism and large domain randomization. Each data point includes **RGB, depth, object pose, camera pose, instance segmentation, 2D bounding box**. [[Google Drive]](https://drive.google.com/drive/folders/1s4pB6p4ApfWMiMjmTXOFco8dHbNXikp-?usp=sharing).

<img src="assets/train_data_vis.png" width="80%">

- To parse the camera params including extrinsics and intrinsics
  ```
  glcam_in_cvcam = np.array([[1,0,0,0],
                          [0,-1,0,0],
                          [0,0,-1,0],
                          [0,0,0,1]]).astype(float)
  W, H = camera_params["renderProductResolution"]
  with open(f'{base_dir}/camera_params/camera_params_000000.json','r') as ff:
    camera_params = json.load(ff)
  world_in_glcam = np.array(camera_params['cameraViewTransform']).reshape(4,4).T
  cam_in_world = np.linalg.inv(world_in_glcam)@glcam_in_cvcam
  world_in_cam = np.linalg.inv(cam_in_world)
  focal_length = camera_params["cameraFocalLength"]
  horiz_aperture = camera_params["cameraAperture"][0]
  vert_aperture = H / W * horiz_aperture
  focal_y = H * focal_length / vert_aperture
  focal_x = W * focal_length / horiz_aperture
  center_y = H * 0.5
  center_x = W * 0.5

  fx, fy, cx, cy = focal_x, focal_y, center_x, center_y
  K = np.eye(3)
  K[0,0] = fx
  K[1,1] = fy
  K[0,2] = cx
  K[1,2] = cy
  ```



# Notes
Due to the legal restrictions of Stable-Diffusion that is trained on LAION dataset, we are not able to release the diffusion-based texture augmented data, nor the pretrained weights using it. We thus release the version without training on diffusion-augmented data. Slight performance degradation is expected.

# Acknowledgement

We would like to thank Jeff Smith for helping with the code release; NVIDIA Isaac Sim and Omniverse team for the support on synthetic data generation; Tianshi Cao for the valuable discussions. Finally, we are also grateful for the positive feebacks and constructive suggestions brought up by reviewers and AC at CVPR.

<img src="assets/cvpr_review.png" width="100%">


# License
The code and data are released under the NVIDIA Source Code License. Copyright © 2024, NVIDIA Corporation. All rights reserved.


# Contact
For questions, please contact [Bowen Wen](https://wenbowen123.github.io/).
