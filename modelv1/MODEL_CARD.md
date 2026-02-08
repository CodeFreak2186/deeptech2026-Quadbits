# Model Card: Wafer Defect Detection Model

## Model Details

**Model Name:** modelv1  
**Model Version:** 1.0  
**Model Type:** Image Classification  
**Architecture:** MobileNetV2 (Transfer Learning)  
**Framework:** PyTorch  
**Date:** February 2026

### Model Description

This is a MobileNetV2-based image classification model fine-tuned for automated wafer defect detection in semiconductor manufacturing. The model identifies 8 different types of wafer defects from grayscale images.

**Developed by:** [Your Organization/Name]  
**Model Architecture:** MobileNetV2 with frozen backbone features  
**Input:** 224x224 grayscale images (converted to 3-channel)  
**Output:** 8-class classification (Bridge, Clean, Etch, Open, Other, Particle, Scratch, Thermal)

## Intended Use

### Primary Use Cases
- Automated quality control in semiconductor manufacturing
- Wafer surface defect detection and classification
- Manufacturing process monitoring and optimization
- Pre-screening before detailed inspection

### Intended Users
- Semiconductor manufacturing facilities
- Quality assurance teams
- Process engineers
- Automated inspection systems

### Out-of-Scope Use Cases
- Medical imaging
- General-purpose object detection
- Real-time critical safety applications without human oversight
- Defect types not represented in the training classes

## Training Data

### Dataset Overview
- **Total Images:** 5,287 wafer defect images
- **Image Type:** Grayscale semiconductor wafer inspection images
- **Input Size:** 224×224 pixels
- **Color Channels:** Grayscale converted to 3-channel RGB

### Class Distribution

| Class    | Train | Valid | Test  | Total | Description                          |
|----------|-------|-------|-------|-------|--------------------------------------|
| Open     | 707   | 153   | 151   | 1,011 | Broken lines, open circuits          |
| Particle | 700   | 150   | 150   | 1,000 | Contamination, particle defects      |
| Scratch  | 700   | 150   | 150   | 1,000 | CMP scratches, surface scratches     |
| Etch     | 591   | 128   | 126   | 845   | Block etch, incomplete etch patterns |
| Clean    | 545   | 118   | 116   | 779   | Clean/normal wafer surfaces          |
| Other    | 287   | 63    | 61    | 411   | Miscellaneous defects                |
| Bridge   | 92    | 21    | 19    | 132   | Shorts, metal bridges                |
| Thermal  | 76    | 17    | 16    | 109   | SEZ burnt, heat damage               |

**Note:** Dataset is imbalanced, reflecting real-world semiconductor defect distributions.

### Data Split
- **Training Set:** 3,698 images (69.9%)
- **Validation Set:** 800 images (15.1%)
- **Test Set:** 789 images (14.9%)

### Data Preprocessing
- Grayscale conversion to 3-channel
- Resize to 224×224 pixels
- Normalization: mean=0.5, std=0.5
- **Thermal class augmentation:** Random rotation (±5°) and horizontal flip

### Class Balancing
- **WeightedRandomSampler** used during training
- Class weights computed inversely proportional to class frequency
- Balanced sampling with target of 76 samples per class per epoch

## Training Procedure

### Hyperparameters
- **Base Model:** MobileNetV2 (pretrained on ImageNet)
- **Training Strategy:** Transfer learning with frozen backbone
- **Optimizer:** Adam (learning rate: 1e-3)
- **Loss Function:** CrossEntropyLoss with class weights
- **Batch Size:** 8
- **Epochs:** 12
- **Device:** CPU
- **Threads:** 4
- **Image Size:** 224×224

### Training Configuration
- **Frozen Layers:** All feature extraction layers (backbone)
- **Trainable Layers:** Final classifier layer only
- **Final Layer:** Linear(1280 → 8 classes)

## Performance Metrics

### Overall Performance

| Metric            | Value  |
|-------------------|--------|
| **Test Accuracy** | 65.3%  |
| **Best Val Acc**  | 67.6%  |
| **Final Val Acc** | 64.6%  |

### Training Progress

| Epoch | Training Loss | Validation Accuracy |
|-------|---------------|---------------------|
| 1     | 0.9899        | 44.8%               |
| 2     | 0.7288        | 57.4%               |
| 3     | 0.6051        | 49.4%               |
| 4     | 0.5402        | 57.8%               |
| 5     | 0.4440        | 56.1%               |
| 6     | 0.5285        | 66.1%               |
| 7     | 0.4653        | 62.5%               |
| 8     | 0.4282        | 67.1%               |
| 9     | 0.4300        | **67.6%** ← Best    |
| 10    | 0.5148        | 62.5%               |
| 11    | 0.4150        | 67.5%               |
| 12    | 0.5306        | 64.6%               |

### Performance Characteristics
- Model achieves reasonable performance given dataset imbalance
- Minority classes (Bridge, Thermal) may have lower individual accuracy
- Best validation accuracy: 67.6% at epoch 9
- Test accuracy of 65.3% indicates some generalization capability

## Model Files

| File              | Format | Size | Description                                |
|-------------------|--------|------|--------------------------------------------|
| `modelv1.pth`     | PyTorch| -    | PyTorch model weights (state_dict)         |
| `modelv1.onnx`    | ONNX   | -    | ONNX format for deployment and inference   |

### ONNX Model Specifications
- **Opset Version:** 11
- **Input Name:** "input"
- **Output Name:** "output"
- **Dynamic Axes:** Batch size
- **Optimization:** Constant folding enabled
- **Target Platform:** CPU inference, Edge deployment (NXP eIQ compatible)

## Limitations

### Technical Limitations
1. **Modest Accuracy:** 65.3% test accuracy may not be sufficient for production use without human verification
2. **Class Imbalance:** Performance likely lower on minority classes (Bridge, Thermal)
3. **Generalization:** Trained on specific dataset; may not generalize to different equipment or processes
4. **CPU-Optimized:** Model optimized for CPU inference, not GPU

### Known Issues
- Validation accuracy shows some instability across epochs
- Training loss doesn't consistently decrease, suggesting possible:
  - Learning rate may need tuning
  - Data augmentation could be enhanced
  - Model capacity vs. data complexity mismatch

### Recommended Mitigations
- Use as pre-screening tool with human verification
- Collect more data for minority classes
- Implement confidence thresholding
- Regular retraining with new production data

## Ethical Considerations

### Potential Biases
- Dataset may reflect biases from specific manufacturing equipment or processes
- Imbalanced class distribution may lead to bias toward majority classes
- Geographic or equipment-specific biases possible

### Responsible Use
- Should not replace human inspection entirely
- Results should be reviewed by qualified personnel
- Regular monitoring for performance degradation
- Document any incidents of misclassification

## Recommendations

### For Best Performance
1. Use confidence thresholds (e.g., flag predictions below 70% confidence for review)
2. Monitor per-class performance metrics in production
3. Retrain model periodically with new defect examples
4. Augment dataset for minority classes (Bridge, Thermal)
5. Consider ensemble methods or more sophisticated architectures

### Model Improvement Suggestions
- Collect more samples for minority classes
- Experiment with different augmentation strategies
- Try unfreezing more layers during fine-tuning
- Test larger architectures (EfficientNet, ResNet)
- Implement focal loss for better class imbalance handling
- Increase training epochs with learning rate scheduling

## Technical Specifications

### Input Specification
```python
Input Shape: (batch_size, 3, 224, 224)
Data Type: float32
Value Range: [-1.0, 1.0] (normalized)
Color Space: Grayscale (replicated to 3 channels)
```

### Output Specification
```python
Output Shape: (batch_size, 8)
Data Type: float32
Format: Logits (raw scores before softmax)
Classes: ['Bridge', 'Clean', 'Etch', 'Open', 'Other', 'Particle', 'Scratch', 'Thermal']
```

### Inference Example
```python
import torch
from torchvision import models, transforms
from PIL import Image

# Load model
model = models.mobilenet_v2()
model.classifier[1] = torch.nn.Linear(1280, 8)
model.load_state_dict(torch.load('modelv1.pth'))
model.eval()

# Preprocess image
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

image = Image.open('wafer_image.jpg')
input_tensor = transform(image).unsqueeze(0)

# Inference
with torch.no_grad():
    output = model(input_tensor)
    probabilities = torch.nn.functional.softmax(output, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1)

classes = ['Bridge', 'Clean', 'Etch', 'Open', 'Other', 'Particle', 'Scratch', 'Thermal']
print(f"Predicted: {classes[predicted_class.item()]} ({probabilities[0][predicted_class].item():.2%})")
```

## Citation

If you use this model, please cite:

```bibtex
@misc{wafer_defect_model_v1,
  title={Wafer Defect Detection Model v1.0},
  author={[Your Name/Organization]},
  year={2026},
  note={MobileNetV2-based transfer learning model for semiconductor wafer defect classification}
}
```

## Contact & Support

For questions, issues, or feedback:
- **Maintainer:** [Your Name/Email]
- **Repository:** [GitHub URL if applicable]
- **Last Updated:** February 2026

## License

[Specify your license here - e.g., MIT, Apache 2.0, Proprietary, etc.]

## Changelog

### Version 1.0 (February 2026)
- Initial release
- MobileNetV2 architecture with transfer learning
- 8-class wafer defect classification
- 65.3% test accuracy
- ONNX export for deployment
