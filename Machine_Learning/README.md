# StyloSense
Bangkit Final Project
Project Name : StyloSense
Description
In our project, we use MTCNN to generate bounding boxes around faces in the input images. Once the bounding box is identified, we crop the image to focus on the face region. This cropped image is then used for skin tone classification.

The skin tone classification model categorizes the skin tone into one of four classes: putih, kuning, coklat, and hitam. Based on the predicted skin tone, we recommend suitable outfit colors as follows:

- Putih: Bright red, Dark brown, Bright golden yellow, Black, Navy blue
- Kuning: Light blue, Light pink, Mint green, Lavender, Peach
- Coklat: Maroon, Mustard yellow, Olive green, Coral, Light brown
- Hitam: Light beige, Gold, Silver, Royal blue, Magenta

This approach not only helps in identifying the skin tone accurately but also enhances the user's experience by providing personalized outfit color recommendations.

This is step to step to run the app on your local environment:


1. After clone our repository, configure the flask app 
```
export FLASK_APP=app.py
```

2. Install the environment
```
pip install Flask>=2.2.2
pip install Pillow==9.5.0
pip install mtcnn==0.1.1
pip install opencv-python==4.7.0.72
pip install opencv-python-headless==4.10.0.82
pip install tensorflow==2.12.0
pip install streamlit==1.31.1
pip install streamlit-webrtc==0.47.7

```

3. Run our app!
```
flask run
```

4. Open your browser at:
```
https://127.0.0.1:5000/
```

Finish, you can try uploading your image!

Dataset Used: 
1. https://www.kaggle.com/datasets/jessicali9530/celeba-dataset
2. https://skintone.google/mste-dataset

   
For more detailed instructions, you can check these notebooks:

Extracting faces from image: model_cropped_image.ipynb

skin tone classification: image_classification.ipynb
