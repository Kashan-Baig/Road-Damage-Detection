# Road Damage Detection using YOLO11

An object detection project using **YOLO11n** to detect road damage, specifically potholes and manholes, in images and videos.

## Classes

| ID | Class |
|---|---|
| 0 | Pothole |
| 1 | Manhole |

## Technologies
- Python
- YOLO11n (Ultralytics)
- PyTorch
- OpenCV
- Google Colab
- Matplotlib

## Dataset

The dataset contains two classes: potholes and manholes.

| Split | Potholes | Manholes |
|---|---:|---:|
| Train | 987 | 756 |
| Validation | 135 | 95 |
| Test | 139 | 105 |

## Training Configuration

| Parameter | Value |
|---|---|
| Model | YOLO11n |
| Pretrained weights | yolo11n.pt |
| Epochs | 100 |
| Batch size | 8 |
| Image size | 768 |
| Patience | 15 |
| GPU | NVIDIA Tesla T4 |
| Best epoch | 72 |
| Training stopped | Epoch 87 |

## Model Evaluation

### 768 × 768 Resolution

| Metric | Validation | Test |
|---|---:|---:|
| Precision | 0.651 | 0.668 |
| Recall | 0.579 | 0.570 |
| mAP50 | 0.614 | 0.5913 |
| mAP50-95 | 0.283 | 0.2576 |

### Resolution Comparison

| Metric | 640 × 640 | 768 × 768 |
|---|---:|---:|
| Precision | 0.656 | 0.651 |
| Recall | 0.613 | 0.579 |
| mAP50 | 0.629 | 0.614 |
| mAP50-95 | 0.295 | 0.283 |

**Observation:** Both resolutions produced similar results. The 640 × 640 model performed slightly better on the reported validation metrics.

### Class-wise Test Results (768)

| Class | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| Pothole | 0.597 | 0.360 | 0.414 | 0.154 |
| Manhole | 0.738 | 0.781 | 0.768 | 0.362 |

## Additional Experiments

The following experiments and analyses were also performed:

- Comparison of 640 and 768 image resolutions.
- Testing different confidence thresholds (0.10, 0.15, 0.25, 0.40 and 0.50).
- Class distribution analysis.
- Bounding box area analysis.
- Visual inspection of training annotations.
- Image prediction on unseen test images.
- Video inference using road footage.
- Comparison of pothole and manhole detection performance.

## Key Observations

- Manhole detection performs better than pothole detection.
- Pothole recall remains an area for improvement.
- Both classes contain many small objects.
- Class imbalance is relatively mild.
- Annotation consistency and dataset diversity may affect detection performance.
- Increasing resolution from 640 to 768 did not improve the reported validation results.

## Installation

```bash
git clone YOUR_REPOSITORY_URL
cd RoadDamageDetection

python -m venv venv
venv\Scripts\activate

pip install ultralytics opencv-python matplotlib numpy
```

## Prediction

Run your prediction script:

```bash
python predict.py
```

For video inference:

```bash
python test_video.py
```

## Future Improvements

- Increase pothole detection recall.
- Improve annotation quality.
- Add more diverse road images.
- Experiment with YOLO11s.
- Evaluate on more real-world road videos.

## Conclusion

This project implements a YOLO11-based road damage detection system for potholes and manholes. Different image resolutions, confidence thresholds, dataset characteristics, and image/video predictions were evaluated to understand model performance and identify areas for improvement.