from flask import Flask, request, jsonify, render_template
import os
import logging

import torch
import torch.nn as nn

import timm
from torchvision import models
from torchvision import transforms

from PIL import Image
import numpy as np

import cv2
import base64

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from flask_cors import CORS

from flask import send_from_directory

# APP CONFIG

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "saved_models")

device = torch.device("cpu")

ALLOWED_MODELS = ["EfficientNet", "DenseNet121", "MobileNetV3"]
classes = ["NORMAL", "PNEUMONIA"]

validator_classes = [
    "CHEST_XRAY",
    "NOT_CHEST_XRAY"
]

# ==========================================
# CHEST X-RAY VALIDATOR
# ==========================================

validator_model = models.mobilenet_v3_small(
    weights=None
)

validator_model.classifier[3] = nn.Linear(
    validator_model.classifier[3].in_features,
    2
)

validator_model.load_state_dict(
    torch.load(
        os.path.join(
            MODEL_DIR,
            "chest_validator.pth"
        ),
        map_location=device
    )
)

validator_model.eval()

print("✅ Chest Validator loaded")

# Home
FRONTEND_DIR = os.path.join(BASE_DIR, "..", "frontend")

@app.route("/")
def serve_frontend():
    return send_from_directory(FRONTEND_DIR, "index.html")

@app.route("/<path:path>")
def static_files(path):
    return send_from_directory(FRONTEND_DIR, path)
# TRANSFORM

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# LOAD MODELS

# EfficientNet
efficientnet_model = timm.create_model("efficientnet_b0", pretrained=False)
efficientnet_model.classifier = nn.Linear(
    efficientnet_model.classifier.in_features, 2
)
efficientnet_model.load_state_dict(
    torch.load(os.path.join(MODEL_DIR, "efficientnet.pth"), map_location=device)
)
efficientnet_model.eval()

# DenseNet121
densenet_model = models.densenet121(pretrained=False)
densenet_model.classifier = nn.Linear(
    densenet_model.classifier.in_features, 2
)
densenet_model.load_state_dict(
    torch.load(os.path.join(MODEL_DIR, "densenet121.pth"), map_location=device)
)
densenet_model.eval()

# MobileNetV3
mobilenet_model = models.mobilenet_v3_small(pretrained=False)
mobilenet_model.classifier[3] = nn.Linear(
    mobilenet_model.classifier[3].in_features, 2
)
mobilenet_model.load_state_dict(
    torch.load(os.path.join(MODEL_DIR, "mobilenetv3.pth"), map_location=device)
)
mobilenet_model.eval()

print("✅ All models loaded")

def validate_chest_xray(image):

    image = image.convert("RGB")

    image = transform(image).unsqueeze(0)

    validator_model.eval()

    with torch.no_grad():

        outputs = validator_model(image)

        probs = torch.softmax(
            outputs,
            dim=1
        )

        confidence, predicted = torch.max(
            probs,
            1
        )

    return {
        "class": validator_classes[
            predicted.item()
        ],
        "confidence": float(
            confidence.item()
        )
    }

# PREDICTION FUNCTION

def predict_image(image, model_name):

    image = image.convert("RGB")
    image = transform(image).unsqueeze(0)

    if model_name == "EfficientNet":
        model = efficientnet_model
    elif model_name == "DenseNet121":
        model = densenet_model
    else:
        model = mobilenet_model

    model.eval()

    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)

    return {
        "prediction": classes[predicted.item()],
        "confidence": float(confidence.item())
    }

# GRAD-CAM FUNCTION

def generate_gradcam(image, model_name):

    image_rgb = image.convert("RGB")
    img_resized = image_rgb.resize((224, 224))

    rgb_array = np.array(img_resized) / 255.0
    input_tensor = transform(image_rgb).unsqueeze(0)

    if model_name == "EfficientNet":
        model = efficientnet_model
        target_layer = model.conv_head

    elif model_name == "DenseNet121":
        model = densenet_model
        target_layer = model.features[-1]

    else:
        model = mobilenet_model
        target_layer = model.features[-1]

    model.eval()

    cam = GradCAM(
        model=model,
        target_layers=[target_layer]
    )

    grayscale_cam = cam(input_tensor=input_tensor)[0]

    visualization = show_cam_on_image(
        rgb_array,
        grayscale_cam,
        use_rgb=True
    )

    _, buffer = cv2.imencode('.jpg', visualization)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    return img_base64

# API PREDICT

@app.route("/predict", methods=["POST"])
def predict():

    try:
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        file = request.files["image"]
        model_name = request.form.get("model")

        if model_name not in ALLOWED_MODELS:
            return jsonify({"error": "Invalid model"}), 400

        image = Image.open(file)

        # ==========================================
        # VALIDATION CHEST X-RAY
        # ==========================================

        validation = validate_chest_xray(image)

        if validation["class"] != "CHEST_XRAY":
            return jsonify({
                "error": "Uploaded image is not a Chest X-Ray",
                "validator_prediction": validation["class"],
                "validator_confidence": validation["confidence"]
            }), 400

        # ==========================================
        # PNEUMONIA DETECTION
        # ==========================================

        result = predict_image(image, model_name)

        # ==========================================
        # GRAD-CAM
        # ==========================================

        gradcam_img = generate_gradcam(
            image,
            model_name
        )

        # ==========================================
        # RESPONSE
        # ==========================================

        return jsonify({
            "validator_prediction": validation["class"],
            "validator_confidence": validation["confidence"],
            "prediction": result["prediction"],
            "confidence": result["confidence"],
            "gradcam": gradcam_img
        })

    except Exception as e:
        logging.error(str(e))
        return jsonify({"error": str(e)}), 500

# RUN APP

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)


