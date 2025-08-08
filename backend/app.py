from flask import Flask, request, jsonify, render_template
from flask_cors import CORS  # 1. Import CORS
import predict
import torch
import os

# 2. Initialize Flask and CORS
app = Flask(__name__, static_folder='../frontend', template_folder='../frontend')
CORS(app)  # This enables CORS for all routes

# Load model once at startup
model = predict.load_model()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
        
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    try:
        img_tensor = predict.preprocess_image(file.read())
        digit, probabilities = predict.predict_digit(model, img_tensor)
        
        return jsonify({
            'digit': digit,
            'probabilities': probabilities
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)