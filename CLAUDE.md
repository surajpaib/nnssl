# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

nnssl is a framework for self-supervised learning (SSL) pre-training of 3D medical images. It's designed to work alongside nnU-Net and supports multiple SSL methods including MAE, SimMIM, SimCLR, VoCo, VolumeFusion, Models Genesis, SwinUNETR, and Spark 3D.

The framework supports two main architectures:
- **ResEnc-L**: CNN architecture
- **Primus-M**: Transformer architecture

## Required Environment Variables

Set these before using the framework:

```bash
export nnssl_raw=/path/to/raw/datasets          # Raw pretrain_data.json files
export nnssl_preprocessed=/path/to/preprocessed # Preprocessed data storage
export nnssl_results=/path/to/results           # Model checkpoints and results
```

Optional:
```bash
export nnUNet_n_proc_DA=XX  # Number of data augmentation workers (10-32 depending on GPU)
```

## Common Commands

### Installation
```bash
pip install -e .
```

### Dataset Preparation
```bash
# Convert OpenMind dataset to nnssl format
nnssl_convert_openmind --openmind_root_dir /path/to/OpenMind

# For custom datasets, create a pretrain_data.json using the Collection dataclass
# See src/nnssl/data/raw_dataset.py
```

### Preprocessing Pipeline
```bash
# Complete pipeline (fingerprint + plan + preprocess)
nnssl_plan_and_preprocess -d <Dataset ID>

# Or run steps individually:
nnssl_extract_fingerprint -d <ID> -np 20
nnssl_plan_experiment -d <ID>
nnssl_preprocess -d <ID> -np 12 -c <CONFIG>
```

CONFIG options: `onemmiso` (1mm isotropic), `median` (median spacing), `noresample`

### Training
```bash
# ResEnc-L MAE training (single GPU)
nnssl_train <ID> <CONFIG> -tr BaseMAETrainer -p nnsslPlans

# ResEnc-L MAE training (4 GPUs)
nnssl_train <ID> <CONFIG> -tr BaseMAETrainer_BS8 -p nnsslPlans -num_gpus 4

# Primus-M (Transformer) MAE training
nnssl_train <ID> <CONFIG> -tr BaseEvaMAETrainer -p nnsslPlans

# With Weights & Biases logging
nnssl_train_wandb <ID> <CONFIG> -tr <TrainerName> -p nnsslPlans
```

Common trainer names:
- MAE: `BaseMAETrainer`, `BaseEvaMAETrainer` (Eva variants for transformers)
- SimMIM: `SimMIMEvaTrainer`
- Spark: `SparkTrainer`, `EffSparkTrainer`
- SimCLR: `SimCLRTrainer`, `SimCLREvaTrainer`
- VolumeFusion: `VolumeFusionTrainer`, `VolumeFusionEvaTrainer`
- VoCo: Check `src/nnssl/training/nnsslTrainer/volume_contrastive/`
- Models Genesis: Check `src/nnssl/training/nnsslTrainer/models_genesis/`
- SwinUNETR: `SwinUNETRTrainer`

Append `_BS8`, `_BS32` etc. to trainer names for specific batch sizes per GPU.

### Testing
```bash
python -m pytest tests/
```

## Architecture

### Core Directory Structure

- **src/nnssl/data/**: Dataset handling, raw dataset dataclasses, and data loading
  - `raw_dataset.py`: Collection, Dataset, Subject, Session, Image dataclasses for pretrain_data.json
  - `dataloading/`: Data loaders and batch generators
  - `nnsslFilter/`: Dataset filtering utilities

- **src/nnssl/experiment_planning/**: Fingerprinting, planning, and preprocessing orchestration
  - `dataset_fingerprint/`: Dataset analysis (shape, spacing, intensity)
  - `experiment_planners/`: Plans defining preprocessing strategies

- **src/nnssl/preprocessing/**: Data preprocessing components
  - `cropping/`: Foreground cropping utilities
  - `normalization/`: Intensity normalization
  - `resampling/`: Spatial resampling
  - `preprocessors/`: Main preprocessing logic

- **src/nnssl/ssl_data/**: SSL-specific data components
  - `dataloading/`: Transform-specific data loaders (e.g., voco_transform.py, volume_fusion_transform.py)
  - `data_augmentation/`: Augmentation pipelines for SSL methods

- **src/nnssl/training/**: Training infrastructure
  - `nnsslTrainer/`: All trainer implementations organized by SSL method
    - `AbstractTrainer.py`: Base trainer class
    - `masked_image_modeling/`: MAE, SimMIM, Spark variants
    - `volume_fusion/`: VolumeFusion trainers
    - `volume_contrastive/`: VoCo trainers
    - `simCLR/`: SimCLR trainers
    - `models_genesis/`: Models Genesis trainers
    - `swinunetr_pretrain/`: SwinUNETR trainers
  - `loss/`: Loss functions
  - `lr_scheduler/`: Learning rate schedulers
  - `logging/`: Logging utilities

- **src/nnssl/architectures/**: Network architectures
  - ResEnc variants, EVA (transformer), Spark, VoCo, SwinUNETR
  - `architecture_registry.py`: Central registry for all architectures

- **src/nnssl/model_sharing/**: Checkpoint upload/download to HuggingFace

- **src/nnssl/run/**: Training entry points
  - `run_training.py`: Main training script
  - `load_pretrained_weights.py`: Weight loading utilities

### Key Architectural Concepts

**Dataset Format (pretrain_data.json)**:
- BIDS-like hierarchical structure: Collection → Dataset → Subject → Session → Images
- Supports relative paths with `$` prefix (e.g., `$nnssl_raw/path/to/image.nii.gz`)
- Allows metadata at each level (dataset_info, subject_info, session_info, image_info)
- Associated masks: anonymization_mask and anatomy_mask for each image

**Preprocessing Pipeline**:
1. Fingerprinting: Analyzes raw data (shapes, spacings, intensities)
2. Planning: Determines target spacing and patch size based on fingerprint
3. Preprocessing: Crops, normalizes, resamples, and saves as blosc2 compressed format

**Trainer Structure**:
- All trainers inherit from `AbstractBaseTrainer`
- Trainers define: network architecture, loss function, data loading transforms
- Trainers save adaptation_plan.json for downstream nnU-Net fine-tuning
- Supports multi-GPU training via DDP

**Architecture Selection**:
- ResEnc-L: CNN-based, used with trainers like `BaseMAETrainer`
- Primus-M / EVA: Transformer-based, used with "Eva" trainers like `BaseEvaMAETrainer`
- Architecture automatically determined by trainer or explicitly set in trainer class

## Important Notes

- The framework uses blosc2 compression for fast I/O with partial decompression
- Pre-training does not track metrics beyond train/validation loss (no linear probing for segmentation)
- For downstream tasks, use the forked nnU-Net: https://github.com/TaWald/nnUNet
- Checkpoints include adaptation_plan.json for automatic nnU-Net compatibility
- Trainers with `_test` suffix are for debugging/testing purposes

## Dataset Conversion

To create a custom dataset:
1. Write a script that builds a `Collection` object (see `src/nnssl/data/raw_dataset.py`)
2. Use `.to_dict()` method to generate pretrain_data.json
3. Place in `$nnssl_raw/DatasetXXX_YourName/pretrain_data.json`

Reference: `src/nnssl/data/dataset_conversion/Dataset001_OpenMind.py`

## Pre-commit Hooks

The repository uses pre-commit hooks for:
- YAML validation
- Trailing whitespace removal
- Merge conflict detection

Install hooks: `pre-commit install`
