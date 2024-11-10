# Image Classification with CNN

## Overview
Classify images using Convolutional Neural Networks (CNNs) in Python.

## Project Structure
```
Imageclass-with-CNN/
├── data/                # Datasets
├── models/              # Saved models
├── scripts/             # Training/evaluation scripts
└── requirements.txt     # Dependencies
```

## Setup
Install dependencies:
```bash
pip install -r requirements.txt
```

## Run Instructions
1. Clone the repo:
   ```bash
   git clone https://github.com/Kshreya08/Imageclass-with-CNN.git
   cd Imageclass-with-CNN
   ```
2. Train:
   ```bash
   python scripts/train_model.py
   ```
3. Evaluate:
   ```bash
   python scripts/evaluate_model.py
   ```

## Image Prediction
Load and predict:
```python
model = models.load_model('models/image_classifier.keras')
img = cv.imread('path/to/image.jpg')
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
img = np.array([img]) / 255.0
predicted_class = class_names[np.argmax(model.predict(img))]
print(f'Prediction: {predicted_class}')
```

## Author
[Kshreya08](https://github.com/Kshreya08)
