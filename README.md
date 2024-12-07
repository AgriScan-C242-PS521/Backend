# AgriScan Flask Backend 🌱

This repository contains the backend implementation for the **AgriScan** application, which uses a pre-trained deep learning model to classify tomato leaf diseases. The backend is built using **Flask** and supports image-based predictions.

---

## 🚀 Features

- **Disease Prediction**: Upload an image of a tomato leaf, and the API predicts the type of disease (or healthy status).
- **Pre-trained Model**: A TensorFlow model trained on a dataset of tomato leaf images.
- **REST API**: Simple and efficient endpoints to integrate with the mobile application.

---

## 🛠️ Project Structure

```
Flask_AgriScan/
├── app.py                  # Flask application
├── model/
│   └── tomato_model.h5     # Pre-trained model file
├── templates/              # (Optional) HTML templates for web UI
├── static/                 # (Optional) Static files (e.g., CSS, JS)
└── README.md               # Project documentation
```

---

## ⚙️ Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/Flask_AgriScan.git
cd Flask_AgriScan
```

### 2. Install Dependencies
Ensure you have Python 3.8+ installed. Install required libraries:
```bash
pip install -r requirements.txt
```

### 3. Download the Pre-trained Model
Download the pre-trained model (`tomato_model.h5`) from [this link](https://drive.google.com/file/d/1HHkwlZqO0_oYR7o_T9CreaUNBbp8lbkE/view?usp=drive_link).  
Place it in the `model/` directory:
```
Flask_AgriScan/
├── model/
│   └── tomato_model.h5
```

### 4. Run the Application
Start the Flask server:
```bash
python app.py
```
The server will start on `http://127.0.0.1:5000`.

---

## 📡 API Usage

### **1. Test API Health**
Check if the API is running:
```bash
GET /
```
Response:
```json
"Tomato Leaf Disease Classification API is running!"
```

### **2. Predict Disease**
Endpoint to predict tomato leaf disease:
```bash
POST /predict
```

#### Request
- **Type**: `multipart/form-data`
- **Key**: `file`
- **Value**: Image file (e.g., `test_leaf.jpg`)

#### Example with `cURL`:
```bash
curl -X POST -F "file=@test_leaf.jpg" http://127.0.0.1:5000/predict
```

#### Response
```json
{
  "prediction": [0],
  "confidence": [[0.8, 0.1, 0.1]]
}
```
- `prediction`: Predicted class (index based on your model's labels).
- `confidence`: Confidence scores for each class.

---

## 🧪 Testing the API

You can test the API using:
- **Postman**: Upload an image to the `/predict` endpoint.
- **Python Script**:
```python
import requests

url = 'http://127.0.0.1:5000/predict'
file_path = 'path_to_your_test_image.jpg'

with open(file_path, 'rb') as img_file:
    response = requests.post(url, files={'file': img_file})
print(response.json())
```

---

## 🚀 Deployment

You can deploy this Flask application to any cloud platform, such as:
- **Google Cloud Platform (GCP)**
- **Heroku**
- **AWS Elastic Beanstalk**
- **PythonAnywhere**

---

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributions

Feel free to fork this repository, open issues, or submit pull requests to improve the project.
