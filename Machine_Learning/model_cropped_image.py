# -*- coding: utf-8 -*-
"""model_cropped_image.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1feMIzfbPW98MNVnTlYC-dX9DizE0enr7
"""

from google.colab import drive
drive.mount('/content/drive')

pip install mtcnn

import os
import cv2
from mtcnn import MTCNN
import matplotlib.pyplot as plt

# Inisialisasi MTCNN
mtcnn = MTCNN()

# Direktori sumber gambar
source_folder = '/content/drive/MyDrive/Dataset_Skin_Tone/images/coklat'
# Direktori tujuan untuk menyimpan gambar yang sudah dipotong
destination_folder = '/content/drive/MyDrive/Dataset_Skin_Tone/cropped_images_marcell/coklat'
# Membuat folder tujuan jika belum ada
os.makedirs(destination_folder, exist_ok=True)

# Fungsi untuk memproses gambar dalam folder
def process_images(source_folder, destination_folder):
    for filename in os.listdir(source_folder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(source_folder, filename)
            image = cv2.imread(image_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Deteksi wajah
            faces = mtcnn.detect_faces(image_rgb)

            if faces:
                # Ambil wajah pertama yang terdeteksi
                face = faces[0]
                x, y, width, height = face['box']
                # Memotong gambar sesuai bounding box
                cropped_face = image[y:y+height, x:x+width]

                # Menyimpan gambar hasil potongan
                cropped_face_filename = f"{os.path.splitext(filename)[0]}_face.jpg"
                cropped_face_path = os.path.join(destination_folder, cropped_face_filename)
                cv2.imwrite(cropped_face_path, cropped_face)

                # Gambar bounding box di sekitar wajah yang terdeteksi untuk visualisasi
                cv2.rectangle(image_rgb, (x, y), (x + width, y + height), (0, 255, 0), 4)

                # Tampilkan gambar dengan bounding box
                plt.imshow(image_rgb)
                plt.axis('off')
                plt.show()

# Memproses gambar dalam folder
process_images(source_folder, destination_folder)
