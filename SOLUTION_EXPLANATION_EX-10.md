# Exercise 10: Semantic Segmentation - Solution Explanation

## Table of Contents
1. [Topic Overview](#topic-overview)
2. [Architecture Design](#architecture-design)
3. [Why MobileNetV2?](#why-mobilenetv2)
4. [Training Strategy](#training-strategy)
5. [Key Components Explained](#key-components-explained)
6. [Code Implementation](#code-implementation)

---

## Topic Overview

### What is Semantic Segmentation?

**Semantic Segmentation** is a computer vision task where the goal is to classify **every pixel** in an image into a predefined category. 

| Task | Input | Output |
|------|-------|--------|
| Image Classification | Image | Single label |
| Object Detection | Image | Bounding boxes + labels |
| **Semantic Segmentation** | Image | Dense pixel-wise labels (same size as input) |

### Why Convolutional Layers over Fully-Connected?

1. **Spatial Information Preservation**: Conv layers maintain spatial relationships
2. **Parameter Efficiency**: Shared weights across spatial locations
3. **Variable Input Size**: FCN can handle different image sizes
4. **Translation Invariance**: Features detected anywhere in the image

---

## Architecture Design

### Encoder-Decoder Structure

```
Input Image (N, 3, 240, 240)
        ↓
┌─────────────────────────────┐
│      ENCODER (MobileNetV2)   │
│   Pretrained Feature         │
│   Extractor                  │
│   240×240 → 7×7              │
└─────────────────────────────┘
        ↓
   Features (N, 1280, 7, 7)
        ↓
┌─────────────────────────────┐
│         DECODER              │
│   Transposed Convolutions    │
│   7×7 → 240×240              │
└─────────────────────────────┘
        ↓
Output (N, 23, 240, 240)
```

### Decoder Details

| Layer | Input Size | Output Size | Operation |
|-------|------------|-------------|-----------|
| 1 | (1280, 7, 7) | (256, 15, 15) | ConvTranspose2d + BN + ReLU |
| 2 | (256, 15, 15) | (128, 30, 30) | ConvTranspose2d + BN + ReLU |
| 3 | (128, 30, 30) | (64, 60, 60) | ConvTranspose2d + BN + ReLU |
| 4 | (64, 60, 60) | (32, 120, 120) | ConvTranspose2d + BN + ReLU |
| 5 | (32, 120, 120) | (32, 240, 240) | ConvTranspose2d + BN + ReLU |
| 6 | (32, 240, 240) | (23, 240, 240) | Conv2d 1×1 (classifier) |

---

## Why MobileNetV2?

MobileNetV2 was chosen for several practical reasons:

### 1. Parameter Efficiency (< 5M params requirement)
| Model | Parameters | Fits Constraint? |
|-------|------------|------------------|
| MobileNetV2 | ~3.4M | ✅ Yes |
| ResNet50 | ~25M | ❌ No |
| VGG16 | ~138M | ❌ No |
| AlexNet | ~61M | ❌ No |

### 2. Depthwise Separable Convolutions
Standard convolution is factorized into:
- **Depthwise convolution**: Filters each channel separately
- **Pointwise 1×1 convolution**: Combines channels

This reduces computation by **~8-9x** compared to standard convolutions.

### 3. Inverted Residuals with Linear Bottlenecks
```
Input → Expand (1×1) → Depthwise (3×3) → Project (1×1) → Output
  └──────────────────── Skip Connection ──────────────────┘
```

### 4. Model Size (< 50MB requirement)
- MobileNetV2 encoder: ~14MB
- With decoder: ~20-25MB total
- Well under the 50MB limit

---

## Training Strategy

### Hyperparameters
```python
num_epochs = 50
batch_size = 8
learning_rate = 1e-3
```

### Loss Function
```python
criterion = torch.nn.CrossEntropyLoss(ignore_index=-1, reduction='mean')
```

**Why `ignore_index=-1`?**
- The `void` class (unlabeled pixels) has label `-1`
- These pixels should NOT contribute to the loss
- This prevents the model from learning to predict "unlabeled"

### ReduceLROnPlateau Scheduler

```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='max',      # Reduce when metric stops INCREASING
    factor=0.5,      # New LR = old LR × 0.5
    patience=5,      # Wait 5 epochs before reducing
    verbose=True     # Print when LR changes
)
```

**How it works:**
```
Initial LR: 0.001
     ↓
[Training...]
     ↓
No improvement for 5 epochs
     ↓
LR → 0.0005
     ↓
[Continue training...]
```

**Why use it?**
1. **Adaptive**: Automatically finds when to reduce LR
2. **Escape plateaus**: Lower LR helps fine-tune when stuck
3. **No manual tuning**: Don't need to predefine reduction schedule

### Transfer Learning Strategy
```python
# Freeze early encoder layers (first 50 parameters)
for param in list(self.encoder.parameters())[:50]:
    param.requires_grad = False
```

**Why freeze early layers?**
- Early layers learn generic features (edges, textures)
- These transfer well and don't need retraining
- Prevents overfitting on small dataset (only 276 training images)
- Later layers are fine-tuned for segmentation task

---

## Key Components Explained

### 1. Transposed Convolution (Upsampling)
Unlike regular convolution that reduces spatial size, transposed convolution **increases** it:

```python
nn.ConvTranspose2d(in_channels, out_channels, 
                   kernel_size=3, 
                   stride=2,           # Doubles spatial size
                   padding=1, 
                   output_padding=1)   # Ensures exact doubling
```

### 2. Batch Normalization
```python
nn.BatchNorm2d(num_features)
```
- Normalizes activations across batch
- Stabilizes training
- Allows higher learning rates

### 3. Final Interpolation
```python
if out.shape[2:] != input_size:
    out = nn.functional.interpolate(out, size=input_size, 
                                    mode='bilinear', 
                                    align_corners=False)
```
- Ensures output matches input dimensions exactly
- Handles edge cases where transposed convs don't give exact size

### 4. Evaluation Metric
```python
def evaluate_model(model, dataloader):
    # Only count pixels where target >= 0 (ignore void)
    targets_mask = (targets >= 0)
    accuracy = mean((predictions == targets)[targets_mask])
```

---

## Code Implementation

### Model Architecture (`segmentation_nn.py`)
```python
class SegmentationNN(nn.Module):
    def __init__(self, num_classes=23, hp=None):
        super().__init__()
        
        # Encoder: Pretrained MobileNetV2
        mobilenet = models.mobilenet_v2(weights=...)
        self.encoder = mobilenet.features
        
        # Freeze early layers
        for param in list(self.encoder.parameters())[:50]:
            param.requires_grad = False
        
        # Decoder: 5 upsampling stages + classifier
        self.decoder = nn.Sequential(
            # 5 transposed convolution blocks
            # Final 1×1 conv for classification
        )
    
    def forward(self, x):
        input_size = x.shape[2:]
        features = self.encoder(x)
        out = self.decoder(features)
        # Ensure output matches input size
        if out.shape[2:] != input_size:
            out = F.interpolate(out, size=input_size, ...)
        return out
```

### Training Loop
```python
for epoch in range(num_epochs):
    model.train()
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    
    # Validation
    val_acc = evaluate_model(model, val_loader)
    scheduler.step(val_acc)  # Adjust LR based on val accuracy
```

---

## Summary

| Component | Choice | Reason |
|-----------|--------|--------|
| Encoder | MobileNetV2 | Efficient, < 5M params, good features |
| Decoder | Transposed Conv | Learnable upsampling |
| Loss | CrossEntropy (ignore=-1) | Ignores unlabeled pixels |
| Optimizer | Adam | Adaptive learning rates |
| Scheduler | ReduceLROnPlateau | Automatic LR reduction |
| Transfer Learning | Freeze first 50 params | Prevent overfitting |

**Target Accuracy**: ≥ 64% on test set

---

## References
- [MobileNetV2 Paper](https://arxiv.org/abs/1801.04381)
- [PyTorch Pretrained Models](https://pytorch.org/vision/stable/models.html)
- [Semantic Segmentation Guide](https://www.jeremyjordan.me/semantic-segmentation/)
