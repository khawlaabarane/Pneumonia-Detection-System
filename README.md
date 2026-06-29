# 🩺 Pneumonia Detection from Chest X-ray Images using Deep Learning

## Overview

This project presents an end-to-end Deep Learning solution for the automatic detection of pneumonia from chest X-ray images.

The system combines a Convolutional Neural Network (CNN) for medical image classification with a FastAPI backend and a user-friendly graphical interface, enabling healthcare professionals or researchers to analyze chest X-ray images efficiently.

In addition to binary classification (NORMAL / PNEUMONIA), the application includes an image validation mechanism capable of rejecting unsupported or non-chest X-ray images before inference.

---

## Features

- Automatic pneumonia detection from chest X-ray images.
- Deep Learning model developed with PyTorch.
- Image preprocessing and normalization pipeline.
- REST API built using FastAPI.
- Graphical User Interface (GUI).
- Automatic rejection of invalid or unsupported images.
- Model evaluation on multiple public datasets.
- Easy deployment and inference.

---

## System Architecture

```
Chest X-ray Image
        │
        ▼
 Image Preprocessing
        │
        ▼
 CNN Classification Model
        │
        ▼
Prediction
(NORMAL / PNEUMONIA)
        │
        ▼
 FastAPI Backend
        │
        ▼
 User Interface
```

---

## Dataset

The model was trained and evaluated using publicly available chest X-ray datasets.

Examples include:

- Chest X-ray Pneumonia Dataset
- Additional external datasets for independent validation

*Datasets are not included in this repository due to licensing and size constraints.*

---

## Technologies

- Python
- PyTorch
- FastAPI
- OpenCV
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

---

## Model Development

The project includes:

- Data preprocessing
- Data augmentation
- CNN model training
- Hyperparameter tuning
- Performance evaluation
- Independent validation

Training notebooks are available inside the `notebooks/` directory.

---

## Results

The trained model achieved strong performance on both the testing dataset and external validation datasets.

Evaluation metrics include:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

---

## Screenshots

### Main Interface

*(Insert screenshot here)*

---

### NORMAL Prediction

*(Insert screenshot here)*

---

### PNEUMONIA Prediction

*(Insert screenshot here)*

---

### Invalid Image Rejection

*(Insert screenshot here)*

---

## Repository Structure

```
Pneumonia-Detection/

├── app/
├── models/
├── notebooks/
├── screenshots/
├── requirements.txt
├── README.md
└── LICENSE
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/Pneumonia-Detection.git
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
python app.py
```

or

```bash
uvicorn main:app --reload
```

(depending on your project structure)

---

## Future Improvements

- Vision Transformer (ViT)
- Explainable AI (Grad-CAM)
- Docker deployment
- Cloud deployment
- Multi-class disease classification
- Integration of Large Language Models (LLMs) for medical report generation

---

## Author

**Khawla ABARANE**

Bachelor's Degree in Artificial Intelligence

GitHub: https://github.com/khawlaabarane


---

## License

This project is released under the MIT License.
