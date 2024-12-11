from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
from flask_cors import CORS
from google.cloud import storage
import uuid
import io

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests

# Load your trained model
model = load_model("model/my_model.h5")

# Labels for the model predictions
labels = [
    "bacterial spot",
    "early blight",
    "late blight",
    "leaf moldx",
    "Septoria leaf",
    "two spotted spider mites",
    "target spot",
    "Yellow leaf",
    "mosaic",
    "healthy"
]

# Google Cloud Storage setup
BUCKET_NAME = "tomato-bucket212"

def upload_to_bucket(image_data, filename):
    """Upload an image to Google Cloud Storage."""
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(filename)
    blob.upload_from_string(image_data, content_type="image/jpeg")
    blob.make_public()
    return blob.public_url

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
        
        # Open the image
        img = Image.open(file)
        
        # Save the original image to upload to bucket
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format="JPEG")
        img_byte_arr = img_byte_arr.getvalue()

        # Generate a unique filename and upload to bucket
        unique_filename = f"uploads/{uuid.uuid4()}.jpg"
        image_url = upload_to_bucket(img_byte_arr, unique_filename)
        
        # Preprocess the image for prediction
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
        
        return jsonify({
            "image_url": image_url,
            "top_predictions": top_predictions
        })
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
