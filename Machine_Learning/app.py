from flask import Flask, request, jsonify
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from mtcnn import MTCNN
from PIL import Image
import io

app = Flask(__name__)

# Load the TensorFlow model
model_path = 'modela85va89.h5'
model = load_model(model_path)

# List of classes
classes = ['putih', 'kuning', 'coklat', 'hitam']

# Hex color codes for each class
color_codes = {
    'putih': ['#FF0000', '#654321', '#FFCC33', '#000000', '#000080'],
    'kuning': ['#FFFF00', '#9ACD32', '#FFD700', '#DAA520', '#ADFF2F'],
    'coklat': ['#8B4513', '#A52A2A', '#D2691E', '#CD853F', '#F4A460'],
    'hitam': ['#000000', '#696969', '#808080', '#A9A9A9', '#C0C0C0']
}

# Function to process the image and make predictions
def process_image(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Face detection
    mtcnn = MTCNN()
    faces = mtcnn.detect_faces(image_rgb)

    if faces:
        # Take the first detected face
        face = faces[0]
        x, y, width, height = face['box']
        # Crop the image according to the bounding box
        cropped_face = image_rgb[y:y+height, x:x+width]

        # Process skin tone prediction
        face_resized = cv2.resize(cropped_face, (224, 224))
        face_array = img_to_array(face_resized)
        face_array = np.expand_dims(face_array, axis=0)
        face_array = tf.keras.applications.mobilenet_v3.preprocess_input(face_array)

        # Predict the skin tone class
        prediction = model.predict(face_array)
        predicted_class = classes[np.argmax(prediction)]

        # Add color recommendations based on skin color
        color_description = {
            'putih': 'Bright red, Dark brown, Bright golden yellow, Black, Navy blue',
            'kuning': 'Light blue, Light pink, Mint green, Lavender, Peach',
            'coklat': 'Maroon, Mustard yellow, Olive green, Coral, Light brown',
            'hitam': 'Light beige, Gold, Silver, Royal blue, Magenta'
        }
        recommended_color = color_description.get(predicted_class, 'No recommendation available.')

        # Get the color codes for the predicted class
        hex_codes = color_codes.get(predicted_class, [])

        return predicted_class, recommended_color, hex_codes
    else:
        return "No face detected", "No recommendation available", []

@app.route('/')
def home():
    return "Hello World! Flask API is running."

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected for uploading'}), 400

    try:
        # Process the image
        image = Image.open(io.BytesIO(file.read()))
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Predict the image class
        predicted_class, recommended_color, hex_codes = process_image(image)

        if predicted_class:
            base_url = 'https://storage.googleapis.com/stylosense-ml-bucket/'
            bucket_links = []

            if predicted_class in ['putih', 'kuning', 'coklat', 'hitam']:
                for i in range(1, 6):
                    bucket_links.append(f'{base_url}{predicted_class}/{i}.png')
            else:
                # Handle unexpected predicted_class by returning default images or an error
                return jsonify({'error': 'Unrecognized class'}), 400

            return jsonify({
                'prediction': predicted_class,
                'color recommendation': recommended_color,
                'bucket links': bucket_links,
                'hex codes': hex_codes
            })
        else:
            return jsonify({'error': 'No face detected'}), 400

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
