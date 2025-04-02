# Self-Supervised Object Pose Estimation With Multi-Task-Learning

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration & run](#configuration)
- [Dataset](#contributing)
- [License](#license)

## Overview
- This codebase appears to implement a multi-task geometry-oriented 6D pose estimation framework (MTGOE) for object detection and pose estimation in images. The system combines several key components:

### Multi-task learning architecture:

- Depth estimation
- Semantic segmentation
- Vector field prediction (pointing to object centers)


### Cross-Modal Attention (CMA):

- Connects depth and segmentation branches
- Uses MultiEmbedding for cross-task feature fusion


### Hough Voting mechanism:

- Uses predicted vector fields to vote for object centers
- Estimates 3D translations using depth information


### Self-supervised training:

- Teacher-student framework
- Geometry-guided filtering for reliable pseudo-labels


## Dataset handling:

- BOP dataset integration
- 3D model-based data augmentation

## Requirements
- Python 3.10
- cudatoolkit = 12.4
- pytorch = 2.4

## Code Structure
```
├── config.toml
├── data
│   └── lmo
├── dataset
│   ├── bop_dataset.py
├── main.py
├── model.py
├── REAME.md
├── requirements.txt
├── self_supervision.py
├── TeacherRenderer.py
├── test.py
└── Voting
    ├── HoughVoter.py

```
---
**config.toml**: This config.toml file sets up the training for a deep learning model, likely for computer vision tasks. It defines:
- [training]: How the model learns, including batch size, learning rate, and whether to use self-supervised learning.
- [lambdas]: The importance of different parts of the model's learning goals (like predicting positions, keypoints, etc.).
- [data]: Where the training images are located and their size.

**bop_dataset.py**: to load dataset

**HoughVoter.py**: A CUDA-accelerated PyTorch implementation of Hough voting for 3D object center detection. This class enables differentiable voting operations, making it suitable for end-to-end deep learning pipelines for object detection and pose estimation tasks.

- GPU-accelerated voting using PyTorch tensors
- Differentiable operations supporting backpropagation
- 2D-to-3D center translation using camera parameters
- Vectorized implementation for efficient processing

**main.py**: 
- Main training script for the Multi-Task Geometric Object Estimation (MTGOE) model
- Handles supervised and self-supervised training
- Includes data loading, model initialization, loss computation, and evaluation
- Supports training on different datasets with configurable parameters
- Implements a multi-task loss function for pose estimation, depth prediction, segmentation, and vector field estimation

**model.py**:
- Defines the MTGOE neural network architecture
- Uses MobileNetV2 as the encoder
- Implements Cross-Modal Attention (CMA) modules
- Includes decoders for depth, segmentation, and vector field prediction
- Implements a Hough Voting mechanism for center and translation detection

**self_supervison.py**:
- Implements self-supervised learning techniques
- Contains *GeometryGuidedFilter* for filtering pose predictions
- Includes *TeacherStudentTrainer* for knowledge distillation between teacher and student models
- Provides methods for computing visual and geometric alignment

**TeacherRenderer.py**:
- Implements a rendering class for generating synthetic views of 3D objects
- Uses PyRender for offscreen rendering
- Supports rendering object meshes from different camera poses
- Handles different rendering backends (EGL, OSMesa)

**test.py:**
- Evaluation script for the MTGOE model
- Loads a trained model checkpoint
- Runs inference on a test dataset
- Computes and visualizes various performance metrics
- Generates detailed visualizations of model predictions

**bop_dataset.py**:
- **Note**: In this code just load object_id = 1 need to modify if you want need more data
- Custom PyTorch Dataset for the BOP (Benchmark for 6D Pose Estimation) dataset
- Handles loading of RGB images, masks, and ground truth pose information
- Generates depth maps and vector fields
- Supports different object IDs and dataset splits
- Includes methods for computing object extents and transforming 3D points


## Configuration & run
- Using pip ```install -r requirements.txt```
- **Note**: First need to train with supervised after that to self supervise.
- For training: Edit config, batchsize must 1 after want training, for train using: ```python main.py```
- For test: ```python test.py --checkpoint outputs/best_model.pth --dataset ./data/lmo --obj_id 1 --output_dir test_results```


## Dataset
- **Note**: In this code just load object_id = 1 need to modify if you want need more data
- Download dataset in here: https://bop.felk.cvut.cz/datasets/
```
├── camera.json
├── dataset_info.md
├── lmo_base.zip
├── lmo_models.zip
├── lmo_test_all.zip
├── lmo_test_bop19
│   └── test
│       └── 000002
├── lmo_test_bop19.zip
├── lmo_train.zip
├── models
│   ├── models_info.json
│   ├── obj_000001.ply
│   ├── obj_000005.ply
│   ├── obj_000006.ply
│   ├── obj_000008.ply
│   ├── obj_000009.ply
│   ├── obj_000010.ply
│   ├── obj_000011.ply
│   └── obj_000012.ply
├── models_eval
│   ├── models_info.json
│   ├── obj_000001.ply
│   ├── obj_000005.ply
│   ├── obj_000006.ply
│   ├── obj_000008.ply
│   ├── obj_000009.ply
│   ├── obj_000010.ply
│   ├── obj_000011.ply
│   └── obj_000012.ply
├── README.md
├── test
│   └── 000002
│       ├── depth
│       ├── mask
│       ├── mask_visib
│       ├── rgb
│       ├── scene_camera.json
│       ├── scene_gt_info.json
│       └── scene_gt.json
├── test_targets_bop19.json
└── train
    ├── 000001
    │   ├── depth
    │   ├── mask
    │   ├── mask_visib
    │   ├── rgb
    │   ├── scene_camera.json
    │   ├── scene_gt_info.json
    │   └── scene_gt.json
```


## License
License information for the project.
