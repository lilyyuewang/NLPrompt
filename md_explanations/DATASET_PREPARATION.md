# Dataset Preparation Guide for NLPrompt

This guide explains how to prepare all datasets used in the NLPrompt project to replicate the results.

## Quick Start

1. **Choose your data directory location**:
   - **Option 1 (Recommended)**: Create a folder outside the project (e.g., `~/datasets/` or `/data/datasets/`)
   - **Option 2**: Create a folder inside the project (e.g., `NLPrompt/data/` or `NLPrompt/DATA/`)
   
2. **Set your data directory**: Edit `scripts/nlprompt/main.sh` and set `DATA` to your chosen path:
   - For Option 1: `DATA=/home/yourusername/datasets` or `DATA=~/datasets`
   - For Option 2: `DATA=/home/convex/NLPrompt/data` or `DATA=./data` (relative path)

3. **Create the directory**: `mkdir -p $DATA` (replace `$DATA` with your actual path)

4. **Download datasets**: Follow the instructions below for each dataset you need

5. **Verify structure**: Ensure each dataset follows the expected directory structure

6. **Run training**: The code will automatically create splits and process datasets on first run

## Overview

All datasets should be placed under a common root directory (referred to as `$DATA`). The root directory is specified via the `--root` argument when running training scripts, or set in `scripts/nlprompt/main.sh` as the `DATA` variable.

**Note**: While you can put datasets inside the NLPrompt folder, it's generally recommended to keep them separate because:
- Datasets are large (ImageNet alone is ~150GB, all datasets may require 200GB+)
- Keeping data separate from code makes project management easier
- Easier to share code without datasets (via git)

The expected structure is:
```
$DATA/
├── caltech-101/
├── food-101/
├── oxford_pets/
├── dtd/
├── imagenet/
├── ... (other datasets)
```

## General Setup

1. **Set the data root directory**: Update `DATA=/path/to/datasets` in `scripts/nlprompt/main.sh` or use the `--root` argument when running `train.py`.

2. **Dataset directory naming**: Each dataset expects a specific directory name (see details below). The code will automatically create split files and few-shot data caches in subdirectories.

## Dataset-Specific Instructions

### 1. Caltech-101

**Directory name**: `caltech-101`

**Expected structure**:
```
caltech-101/
└── 101_ObjectCategories/
    ├── airplanes/
    ├── Faces/
    ├── Leopards/
    ├── Motorbikes/
    └── ... (other classes)
```

**Download**: 
- Official website: http://www.vision.caltech.edu/Image_Datasets/Caltech101/
- The dataset will be automatically split into train/val/test sets on first run
- Note: Classes "BACKGROUND_Google" and "Faces_easy" are ignored

---

### 2. Food-101

**Directory name**: `food-101`

**Expected structure**:
```
food-101/
└── images/
    ├── apple_pie/
    ├── baby_back_ribs/
    ├── baklava/
    └── ... (101 food classes)
```

**Download**: 
- Official website: https://data.vision.ee.ethz.ch/cvl/food-101.tar.gz
- Extract the archive and ensure the `images/` folder contains all class folders

---

### 3. Food-101N

**Directory name**: `food-101n` (training) and `food-101` (testing)

**Expected structure**:
```
food-101n/
└── images/
    └── ... (classes with noisy labels)

food-101/
└── images/
    └── ... (clean test set, same as Food-101)
```

**Download**: 
- Food-101N: https://www.kaggle.com/datasets/kuanghueilee/food-101n
- Food-101: Same as above (Food-101N uses Food-101's test set)
- **Important**: Food-101N must be downloaded separately from Kaggle

---

### 4. Oxford-IIIT Pet Dataset

**Directory name**: `oxford_pets`

**Expected structure**:
```
oxford_pets/
├── images/
│   ├── Abyssinian_1.jpg
│   ├── Abyssinian_2.jpg
│   └── ... (all images)
└── annotations/
    ├── trainval.txt
    └── test.txt
```

**Download**: 
- Official website: https://www.robots.ox.ac.uk/~vgg/data/pets/
- Download both images and annotations

---

### 5. Describable Textures Dataset (DTD)

**Directory name**: `dtd`

**Expected structure**:
```
dtd/
└── images/
    ├── banded/
    ├── blotchy/
    ├── braided/
    └── ... (47 texture classes)
```

**Download**: 
- Official website: https://www.robots.ox.ac.uk/~vgg/data/dtd/
- Extract images into the `images/` folder

---

### 6. ImageNet

**Directory name**: `imagenet`

**Expected structure**:
```
imagenet/
├── images/
│   ├── train/
│   │   ├── n01440764/
│   │   ├── n01443537/
│   │   └── ... (1000 classes)
│   └── val/
│       ├── n01440764/
│       ├── n01443537/
│       └── ... (1000 classes)
└── classnames.txt
```

**Download**: 
- Official website: https://www.image-net.org/
- Requires registration and download approval
- Create `classnames.txt` with format: `folder_name class_name` (one per line)

**Note**: ImageNet requires significant storage space (~150GB)

---

### 7. ImageNet-A (Adversarial)

**Directory name**: `imagenet-adversarial`

**Expected structure**:
```
imagenet-adversarial/
├── imagenet-a/
│   └── ... (adversarial images)
└── classnames.txt
```

**Download**: 
- Official repository: https://github.com/hendrycks/natural-adv-examples
- Contains adversarial examples for ImageNet

---

### 8. ImageNet-R (Rendition)

**Directory name**: `imagenet-rendition`

**Expected structure**:
```
imagenet-rendition/
├── imagenet-r/
│   └── ... (rendition images)
└── classnames.txt
```

**Download**: 
- Official repository: https://github.com/hendrycks/imagenet-r
- Contains artistic renditions of ImageNet classes

---

### 9. ImageNet-Sketch

**Directory name**: `imagenet-sketch`

**Expected structure**:
```
imagenet-sketch/
├── images/
│   └── ... (sketch images)
└── classnames.txt
```

**Download**: 
- Official repository: https://github.com/HaohanWang/ImageNet-Sketch
- Contains sketch versions of ImageNet images

---

### 10. ImageNetV2

**Directory name**: `imagenetv2`

**Expected structure**:
```
imagenetv2/
├── matched-frequency/
│   └── ... (or other variant)
└── classnames.txt
```

**Download**: 
- Official repository: https://github.com/modestyachts/ImageNetV2
- Available variants: matched-frequency, thresholded-0.7, top-images

---

### 11. Oxford Flowers-102

**Directory name**: `oxford_flowers`

**Expected structure**:
```
oxford_flowers/
├── jpg/
│   ├── image_00001.jpg
│   ├── image_00002.jpg
│   └── ... (all images)
├── imagelabels.mat
└── cat_to_name.json
```

**Download**: 
- Official website: https://www.robots.ox.ac.uk/~vgg/data/flowers/102/
- Download images and labels
- Create `cat_to_name.json` mapping category IDs to names

---

### 12. Stanford Cars

**Directory name**: `stanford_cars`

**Expected structure**:
```
stanford_cars/
├── cars_train/
│   ├── 00001.jpg
│   ├── 00002.jpg
│   └── ... (training images)
├── cars_test/
│   ├── 00001.jpg
│   └── ... (test images)
└── devkit/
    ├── cars_train_annos.mat
    ├── cars_test_annos_withlabels.mat
    └── cars_meta.mat
```

**Download**: 
- Official website: https://ai.stanford.edu/~jkrause/cars/car_dataset.html
- Download both images and devkit

---

### 13. FGVC Aircraft

**Directory name**: `fgvc_aircraft`

**Expected structure**:
```
fgvc_aircraft/
├── images/
│   └── ... (aircraft images)
├── variants.txt
└── ... (split files)
```

**Download**: 
- Official website: https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/
- Download images and metadata

---

### 14. EuroSAT

**Directory name**: `eurosat`

**Expected structure**:
```
eurosat/
└── 2750/
    ├── AnnualCrop/
    ├── Forest/
    ├── HerbaceousVegetation/
    └── ... (10 land use classes)
```

**Download**: 
- Official repository: https://github.com/phelber/EuroSAT
- Or from: https://zenodo.org/record/7711810
- Extract to have class folders under `2750/`

---

### 15. SUN397

**Directory name**: `sun397`

**Expected structure**:
```
sun397/
├── SUN397/
│   └── ... (scene images organized by class)
└── ClassName.txt
```

**Download**: 
- Official website: https://vision.princeton.edu/projects/2010/SUN/
- Download images and class names file

---

### 16. UCF101

**Directory name**: `ucf101`

**Expected structure**:
```
ucf101/
├── UCF-101-midframes/
│   ├── ApplyEyeMakeup/
│   ├── ApplyLipstick/
│   └── ... (101 action classes)
└── ucfTrainTestlist/
    ├── classInd.txt
    └── ... (split files)
```

**Download**: 
- Official website: https://www.crcv.ucf.edu/data/UCF101.php
- Download videos and extract midframes, or download pre-extracted frames
- Download train/test splits

---

## Automatic Processing

Once datasets are downloaded and placed in the correct directory structure:

1. **First run**: The code will automatically:
   - Create train/val/test splits (if not already present)
   - Save split files as JSON (e.g., `split_zhou_Caltech101.json`)
   - Generate few-shot datasets when `NUM_SHOTS` is specified
   - Cache few-shot data for faster subsequent runs

2. **Subsequent runs**: The code will load preprocessed splits and few-shot data from cache

## Verification

To verify your dataset setup:

1. Check that each dataset directory exists under `$DATA`
2. Run a test training command:
   ```bash
   python train.py \
     --root $DATA \
     --trainer NLPrompt \
     --dataset-config-file configs/datasets/caltech101.yaml \
     --config-file configs/trainers/NLPrompt/rn50.yaml \
     DATASET.NUM_SHOTS 16
   ```

3. The code will print error messages if datasets are missing or incorrectly structured

## Notes

- **Storage requirements**: Ensure you have sufficient disk space. ImageNet alone requires ~150GB, and all datasets together may require 200GB+.
- **Download time**: Some datasets (especially ImageNet) may take significant time to download.
- **Permissions**: Some datasets (like ImageNet, Food-101N) require registration or approval.
- **Split files**: The code uses Zhou et al.'s splits (referenced as "zhou" in split filenames) for consistency with CoOp/CoCoOp baselines.

## Quick Reference: Dataset Directory Names

| Dataset | Directory Name | Notes |
|---------|---------------|-------|
| Caltech-101 | `caltech-101` | Ignore BACKGROUND_Google and Faces_easy |
| Food-101 | `food-101` | Standard food classification |
| Food-101N | `food-101n` | Requires Kaggle download |
| Oxford Pets | `oxford_pets` | Needs annotations folder |
| DTD | `dtd` | Texture dataset |
| ImageNet | `imagenet` | Requires registration |
| ImageNet-A | `imagenet-adversarial` | Adversarial examples |
| ImageNet-R | `imagenet-rendition` | Artistic renditions |
| ImageNet-Sketch | `imagenet-sketch` | Sketch images |
| ImageNetV2 | `imagenetv2` | ImageNet variant |
| Oxford Flowers | `oxford_flowers` | 102 flower classes |
| Stanford Cars | `stanford_cars` | Needs devkit folder |
| FGVC Aircraft | `fgvc_aircraft` | Aircraft classification |
| EuroSAT | `eurosat` | Satellite imagery |
| SUN397 | `sun397` | Scene classification |
| UCF101 | `ucf101` | Action recognition |

## Common Issues and Solutions

### Issue: "Dataset directory not found"
**Solution**: Check that:
- The dataset directory name matches exactly (case-sensitive)
- The directory is under `$DATA` root
- You've set `DATA` correctly in `main.sh` or via `--root` argument

### Issue: "Split file not found" (first run)
**Solution**: This is normal on first run. The code will create the split file automatically. Ensure you have write permissions.

### Issue: "Images not found"
**Solution**: Verify the directory structure matches the expected format. Some datasets require specific subdirectory names (e.g., `images/`, `101_ObjectCategories/`).

## References

- **CoOp repository**: https://github.com/KaiyangZhou/CoOp (for additional dataset preparation details)
- **CoOp DATASETS.md**: https://github.com/KaiyangZhou/CoOp/blob/main/DATASETS.md (original reference from README)
- **NLPrompt paper**: https://arxiv.org/abs/2412.01256

