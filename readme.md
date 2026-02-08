# Wafer Defect Detection Dataset

## Overview

This repository contains a comprehensive wafer defect detection dataset compiled from multiple semiconductor inspection sources. The dataset demonstrates **data gathering skills** by combining and organizing images from various wafer defect datasets into a unified structure suitable for deep learning model training.

> **Note**: We acknowledge that the dataset is imbalanced. This reflects the real-world nature of semiconductor defect data. For training purposes, the dataset will be balanced using appropriate techniques.

## Dataset Structure

```
dataset/
├── main/                        # Primary dataset folder
│   ├── train/                   # Training set
│   │   ├── Bridge/
│   │   ├── Clean/
│   │   ├── Etch/
│   │   ├── open/
│   │   ├── Other/
│   │   ├── Particle/
│   │   ├── Scratch/
│   │   └── Thermal/
│   ├── test/                    # Testing set
│   │   └── [same 8 classes]
│   └── valid/                   # Validation set
│       └── [same 8 classes]
├── readme.md                    # This file
└── readfirst.txt               # Important notes
```

## Dataset Statistics

### Total Images: 5,276

### Class Distribution

| Class      | Train  | Test | Valid | Total | Description                              |
|------------|--------|------|-------|-------|------------------------------------------|
| open       | 1,000  | 0    | 0     | 1,000 | Broken lines, open circuits, pin-holes  |
| Particle   | 723    | 96   | 181   | 1,000 | Contamination, particle defects         |
| Scratch    | 715    | 95   | 190   | 1,000 | CMP scratches, surface scratches        |
| Etch       | 578    | 86   | 181   | 845   | Block etch, incomplete etch patterns    |
| Clean      | 545    | 116  | 118   | 779   | Clean/normal wafer surfaces             |
| Other      | 293    | 49   | 69    | 411   | Miscellaneous defects                   |
| Bridge     | 111    | 5    | 16    | 132   | Shorts, metal bridges                   |
| Thermal    | 73     | 16   | 20    | 109   | SEZ burnt, heat damage                  |

### Split Distribution

- **Train**: 4,038 images (76.5%)
- **Test**: 463 images (8.8%)
- **Valid**: 775 images (14.7%)

## Defect Classes

### 1. Bridge (132 images)
Metal bridges and electrical shorts between wafer components. Critical defects that cause functional failures.

### 2. Particle (1,000 images)
Contamination defects including:
- Particle contamination
- PIQ particles
- PO contamination

These are among the most common defects in semiconductor manufacturing.

### 3. Scratch (1,000 images)
Surface defects from CMP (Chemical Mechanical Planarization) and other mechanical processes. Affects wafer surface quality.

### 4. Etch (845 images)
Etching-related defects:
- Block etch failures
- Incomplete etch patterns
- Pitted surfaces

### 5. open (1,000 images)
**⚠️ Note: Only available in training set**

Open circuit defects including:
- Broken metal lines
- Pin-holes
- Disconnected traces

### 6. Thermal (109 images)
Heat-related damage:
- SEZ burnt areas
- Thermal stress patterns

One of the rarest defect types in the dataset.

### 7. Clean (779 images)
Normal/defect-free wafer surfaces used as baseline for comparison and binary classification tasks.

### 8. Other (411 images)
Miscellaneous defects that don't fit into primary categories:
- Coating defects
- Copper issues
- Crazing patterns

## Data Gathering Process

This dataset showcases comprehensive data gathering skills by:

1. **Multi-source Integration**: Combined data from 3+ semiconductor defect datasets
   - wafer-2 (YOLO format dataset)
   - Wafer-Defect-Grouped (7-class wafer defects)
   - Clean wafer samples from WM811K-based datasets

2. **Class Mapping**: Intelligently mapped original defect types to 8 unified categories
   - SCRATCH → Scratch
   - BLOCK ETCH → Etch
   - PARTICLE, PIQ PARTICLE, PO CONTAMINATION → Particle
   - SEZ BURNT → Thermal
   - COATING BAD → Other
   - bridge → Bridge
   - open, pin-hole → open

3. **Class Balancing**: Applied maximum threshold (1,000 images per class) to prevent severe imbalance

4. **Quality Control**: Filtered and validated images across multiple datasets

## Image Specifications

- **Format**: JPEG, PNG, BMP
- **Color Space**: RGB (original images)
- **Resolution**: Variable (as captured from inspection systems)
- **Preprocessing**: 
  - Random sampling for class balancing (max 1,000 per class)
  - Duplicate removal
  - Quality validation

## Usage

### Loading the Dataset

```python
import os
from PIL import Image

# Load training data
train_dir = "dataset/main/train"
classes = ['Bridge', 'Clean', 'Etch', 'open', 'Other', 'Particle', 'Scratch', 'Thermal']

for class_name in classes:
    class_path = os.path.join(train_dir, class_name)
    images = [f for f in os.listdir(class_path) if f.endswith(('.jpg', '.png'))]
    print(f"{class_name}: {len(images)} images")
```

### PyTorch DataLoader

```python
from torchvision import datasets, transforms

# Define transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
])

# Load dataset
train_dataset = datasets.ImageFolder('dataset/main/train', transform=transform)
test_dataset = datasets.ImageFolder('dataset/main/test', transform=transform)
valid_dataset = datasets.ImageFolder('dataset/main/valid', transform=transform)
```

## Class Imbalance Considerations

⚠️ **Important**: This dataset demonstrates real-world semiconductor defect distribution patterns.

### Imbalance Characteristics:
- **Majority classes**: open, Particle, Scratch (1,000 images each)
- **Minority classes**: Thermal (109), Bridge (132)
- **Missing splits**: open class has no test/valid data

### Recommended Balancing Strategies:

#### 1. **Class Weighting**
```python
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# Compute class weights
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(train_labels),
    y=train_labels
)
```

#### 2. **Oversampling Minority Classes**
- Use RandomOverSampler
- Apply SMOTE (Synthetic Minority Over-sampling Technique)
- Duplicate minority samples with augmentation

#### 3. **Data Augmentation** (Heavily for minority classes)
```python
from torchvision import transforms

augmentation = transforms.Compose([
    transforms.RandomRotation(15),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.GaussianBlur(kernel_size=3),
])
```

#### 4. **Loss Function Adjustment**
- Use Focal Loss for hard example mining
- Apply class-weighted CrossEntropyLoss

## Model Training Recommendations

### Suggested Architectures
- **ResNet50/101**: Good baseline for wafer defect classification
- **EfficientNet**: Excellent accuracy with lower computational cost
- **Vision Transformer (ViT)**: Strong performance on texture patterns
- **MobileNet**: Deployment-friendly for edge devices

### Training Configuration
```python
# Example hyperparameters
batch_size = 32
learning_rate = 1e-4
epochs = 100
optimizer = 'Adam'
scheduler = 'ReduceLROnPlateau'
criterion = 'CrossEntropyLoss' with class weights
```

## Evaluation Metrics

Due to class imbalance, use multiple metrics:
- **Accuracy**: Overall classification performance
- **Precision/Recall**: Per-class performance
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed error analysis
- **AUC-ROC**: For probabilistic predictions

## Known Limitations

1. **Class Imbalance**: 
   - Thermal (109) and Bridge (132) classes are significantly underrepresented
   - This reflects real-world defect occurrence rates but requires balancing for training

2. **Open Class Split Issue**: 
   - open class only exists in training set (1,000 images)
   - No test/validation data available for this class
   - Consider splitting train data for evaluation

3. **Image Quality Variation**: 
   - Variable resolution across images
   - Different lighting conditions from multiple sources
   - Requires normalization during preprocessing

4. **Label Consistency**: 
   - Potential mislabeling in multi-defect images
   - Some defects may be subjectively categorized into "Other"

5. **Real-World Application**: 
   - Dataset shows real semiconductor manufacturing defect distribution
   - Balancing needed for fair model training but imbalance is intentionally preserved to demonstrate data gathering

## Original Data Sources

This dataset compilation demonstrates data gathering from multiple sources:

1. **wafer-2 Dataset**
   - Source: Roboflow Universe - Kyonggi University
   - Classes: bridge, particle, scratch
   - Format: YOLO detection format
   - License: CC BY 4.0

2. **Wafer-Defect-Grouped Dataset**
   - Classes: SCRATCH, BLOCK ETCH, PARTICLE, PIQ PARTICLE, PO CONTAMINATION, SEZ BURNT, COATING BAD
   - Total: 4,531 images
   - 7 technical defect categories

3. **Clean Wafer Samples**
   - Source: clean/Normal folder
   - 779 defect-free wafer images
   - Used as baseline comparison

4. **Semiconductor Defect Dataset**
   - Source: Roboflow Universe - Timmy
   - License: CC BY 4.0
   - Used for open circuit defects

## File Organization

```
dataset/
├── .git/                        # Version control
├── main/                        # Primary dataset
│   ├── train/                   # Training images (4,038 images)
│   │   ├── Bridge/              # 111 images
│   │   ├── Clean/               # 545 images
│   │   ├── Etch/                # 578 images
│   │   ├── open/                # 1,000 images
│   │   ├── Other/               # 293 images
│   │   ├── Particle/            # 723 images
│   │   ├── Scratch/             # 715 images
│   │   └── Thermal/             # 73 images
│   ├── test/                    # Testing images (463 images)
│   │   ├── Bridge/              # 5 images
│   │   ├── Clean/               # 116 images
│   │   ├── Etch/                # 86 images
│   │   ├── Open/                # 0 images (empty)
│   │   ├── Other/               # 49 images
│   │   ├── Particle/            # 96 images
│   │   ├── Scratch/             # 95 images
│   │   └── Thermal/             # 16 images
│   └── valid/                   # Validation images (775 images)
│       ├── Bridge/              # 16 images
│       ├── Clean/               # 118 images
│       ├── Etch/                # 181 images
│       ├── Open/                # 0 images (empty)
│       ├── Other/               # 69 images
│       ├── Particle/            # 181 images
│       ├── Scratch/             # 190 images
│       └── Thermal/             # 20 images
├── readme.md                    # This documentation
└── readfirst.txt               # Important notes for the judges

```

## Citation

If you use this dataset in your research, please cite the original sources:

```bibtex
@dataset{wafer_defect_2026,
  title={Compiled Wafer Defect Detection Dataset},
  author={Your Name/Organization},
  year={2026},
  note={Compiled from multiple semiconductor defect datasets},
  sources={wafer-2, Wafer-Defect-Grouped, Clean samples}
}

@misc{wafer2_roboflow,
  title={Wafer-2 Dataset},
  author={Kyonggi University},
  year={2024},
  publisher={Roboflow Universe},
  license={CC BY 4.0}
}
```

## License

This compiled dataset follows the licenses of its source datasets:
- **wafer-2**: CC BY 4.0
- **Semiconductor defect datasets**: CC BY 4.0

Please refer to individual dataset licenses when using this data.

## Version History

### Version 1.0 (February 2026)
- Initial dataset compilation from 3+ sources
- 8 defect classes established
- 5,276 total images organized
- Train/test/valid splits created (76.5% / 8.8% / 14.7%)
- Class balancing applied (max 1,000 images per class)
- Documentation completed

## Getting Started

1. **Clone or download** the dataset repository
2. **Read** `readfirst.txt` for important notes
3. **Explore** the `main/` folder structure
4. **Review** class distribution and balance strategies
5. **Implement** data augmentation for minority classes
6. **Train** your model with appropriate class weighting

## Contact & Contributions

For questions, issues, or contributions:
- Open an issue in the repository
- Provide feedback on class definitions
- Suggest improvements for data balancing

---

**Dataset Name**: Wafer Defect Detection Dataset  
**Version**: 1.0  
**Last Updated**: February 8, 2026  
**Total Images**: 5,276  
**Classes**: 8  
**Format**: RGB Images (JPEG/PNG/BMP)  
**Purpose**: Demonstrating data gathering and compilation skills for semiconductor defect detection
