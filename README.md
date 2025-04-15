This project implements a machine learning classifier that distinguishes between cat and dog images using SVM (which is less accurate than CNN)

✨ Interactive Version (using streamlit):

I've deployed an interactive version of the code at -
(has drag and drop instead of path input for image)

I uploaded both the original and streamlit version of the code on this repository. The files requirements.txt and psh.py are just extras used for deployment.
The folder pet_photos is used as training data.


🐱🐶 Project Features:

- Converts colored images to grayscale (averages RGB channels) and resizes to 64x64 pixels (but might skip important pixels)

- Allows users to input image paths for classification
  
- Provides confidence scores for predictions
  
- Skips invalid images and handles user input errors
  

📊 How It Works:

1. (Training Phase):
   
Loads labeled images from pet_photos folder

Converts all images to 64x64 grayscale (all images need to be the same size in order to train the model)

Trains SVM classifier on pixel values for each image


2. (Prediction Phase):
   
Processes uploaded image the same way as processing training images

Uses trained model to predict cat/dog class

Returns prediction with confidence score

💡 Notes:

- May not give correct classification results due to skipping important features for image resizing
  
