from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests

# Load your trained model
model = load_model("model/model.h5")

# Labels for the model predictions
labels = [
    "bacterial spot",
    "early blight",
    "Septoria leaf",
    "mosaic",
    "healthy",
    "leaf moldx",
    "target spot",
    "Yellow leaf",
    "two spotted spider mites",
    "late blight"
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if the file part is in the request
        if 'file' not in request.files:
            return jsonify({"error": "No file part in the request"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        # Open and preprocess the image
        img = Image.open(file)
        img = img.resize((224, 224))  # Resize image to match model input size
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array = img_array / 255.0  # Normalize the image
        
        # Make prediction
        prediction = model.predict(img_array)
        top_3_indices = np.argsort(prediction[0])[-3:][::-1]  # Get top 3 predictions
        top_predictions = [
            {"label": labels[i], "confidence": float(prediction[0][i])}
            for i in top_3_indices
        ]
        
        return jsonify({"top_predictions": top_predictions})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
