import torch
from flask import Flask, request, jsonify
from modules import *

app = Flask(__name__)

model = load_model()
device = torch.device('cpu')    
model.to(device)

@app.route('/predict', methods=['POST'])
def predict_():
    if 'image' not in request.json:
        return jsonify({'error': 'No image provided'}), 400

    base64_image = request.json['image']
    print(base64_image)
    try:
        predicted_class = predict(base64_image)

        return jsonify({'prediction': predicted_class})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
