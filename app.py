# app.py
import cv2
import pickle
import numpy as np
from flask import Flask, request, send_file
from io import BytesIO
from PIL import Image

app = Flask(__name__)

# Load parking positions based on the pickle file
def load_parking_positions(pickle_file):
    with open(pickle_file, 'rb') as f:
        return pickle.load(f)

# Function to check parking spaces
def checkParkingSpace(imgPro, img, posList, threshold, scale, text_offset):
    spaceCounter = 0
    for pos in posList:
        x, y, w, h, orientation = pos
        imgCrop = imgPro[y:y + h, x:x + w]
        count = cv2.countNonZero(imgCrop)
        color = (0, 255, 0) if count < threshold else (0, 0, 255)
        thickness = 1
        if count < threshold:
            spaceCounter += 1
        cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)
        cv2.putText(img, str(count), (x, y + h - 3), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1)

    cv2.putText(img, f'Free: {spaceCounter}/{len(posList)}', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 200, 0), 2)

# Process image function
def process_image(file_stream):
    img = Image.open(file_stream)
    img = np.array(img.convert('RGB'))  # Convert to OpenCV format

    # Example: Determine settings based on file name or another attribute
    # Here you should replace this logic with appropriate checks as per your requirements
    image_path = file_stream.filename

    if '1' in image_path:
        pickle_file = 'CarParkPos1'
        threshold = 300
        scale = 1.5
        text_offset = 20
    elif '2' in image_path:
        pickle_file = 'CarParkPos2'
        threshold = 800
        scale = 2
        text_offset = 30
    elif '3' in image_path:
        pickle_file = 'CarParkPos3'
        threshold = 200
        scale = 1
        text_offset = 15
    else:
        raise ValueError("Unrecognized image file")

    posList = load_parking_positions(pickle_file)

    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (3, 3), 1)
    imgThreshold = cv2.adaptiveThreshold(imgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 16)
    imgMedian = cv2.medianBlur(imgThreshold, 5)
    kernel = np.ones((3, 3), np.uint8)
    imgDilate = cv2.dilate(imgMedian, kernel, iterations=1)

    checkParkingSpace(imgDilate, img, posList, threshold, scale, text_offset)

    _, img_encoded = cv2.imencode('.png', img)
    return BytesIO(img_encoded)

@app.route('/')
def index():
    return send_file('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part", 400
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400
    processed_image = process_image(file)
    return send_file(processed_image, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
