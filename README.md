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

<img width="949" height="372" alt="image" src="https://github.com/user-attachments/assets/fe009582-8a69-4b2b-8519-10b51002a331" />


---

### NORMAL Prediction

<img width="949" height="812" alt="image" src="https://github.com/user-attachments/assets/f863a2a3-177b-4142-a8ae-156ef04b9846" />


---

### PNEUMONIA Prediction

<img width="949" height="700" alt="image" src="https://github.com/user-attachments/assets/9d69b4cc-f431-47db-9bcf-622ebef6db2c" />


---

### Invalid Image Rejection


<img width="949" height="330" alt="image" src="https://github.com/user-attachments/assets/88fc5b24-432c-425a-82f3-2e74513affda" />

---

## Repository Structure


Pneumonia-Detection/
│
├── backend/
│   ├── saved_models/
│   │   ├── chest_validator.pth
│   │   ├── densenet121.pth
│   │   ├── efficientnet.pth
│   │   └── mobilenetv3.pth
│   └── app.py
│
├── frontend/
│   ├── index.html
│   ├── script.js
│   └── style.css
│
├── Procfile
├── requirements.txt
└── README.md

```

### Directory Description

- **backend/** – FastAPI backend and trained deep learning models.
- **saved_models/** – Pre-trained models used for pneumonia detection and chest X-ray validation.
- **frontend/** – Web interface built with HTML, CSS and JavaScript.
- **app.py** – Main application entry point.
- **Procfile** – Deployment configuration.
- **requirements.txt** – Python dependencies.
- **README.md** – Project documentation.
```


---

## Installation

Clone the repository:

```bash
git clone https://github.com/khawlaabarane/Pneumonia-Detection-System.git
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
python app.py
```


---

## Future Improvements

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
