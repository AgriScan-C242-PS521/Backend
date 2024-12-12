import requests

url = 'http://127.0.0.1:5000/predict'
file_path = 'test\WhatsApp Image 2024-12-02 at 5.13.15 PM.jpeg'

with open(file_path, 'rb') as img_file:
    response = requests.post(url, files={'file': img_file})

print(response.json())
