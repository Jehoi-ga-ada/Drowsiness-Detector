import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
import base64
from ultralytics import YOLO

def load_model():
    model = YOLO("best.pt")
    return model

def predict(base64_image, device='cpu'):
    model = load_model()
    input_image = preprocess_base64_image(base64_image)
    results = model.predict(source=input_image,
                        save=False, conf=0.1, iou=0.5)
    names = model.names
    try:
        predicted_class = names[int(results[0].boxes.cls[0])]
    except:
        predicted_class = 'empty'
    print(predicted_class)
    return predicted_class

def preprocess_base64_image(base64_string):
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data))
    image.save('buffer.jpg', 'JPEG')
    return image
