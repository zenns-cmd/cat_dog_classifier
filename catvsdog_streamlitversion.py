import os
import numpy as np
from sklearn.svm import SVC
from matplotlib.image import imread
from pathlib import Path
import streamlit as st


st.set_page_config(page_title="Cat/Dog Classifier", page_icon="🐾") 


st.markdown("<h1 style='text-align: center;'>🌟 Interactive Cat/Dog Classifier </h1>", unsafe_allow_html=True) 


if "model" not in st.session_state: #initializes session states
    st.session_state.model = None
    st.session_state.training_complete = False
    st.session_state.x = None
    st.session_state.y = None

training_data = Path("pet_photos")

def load_images():
    st.write("\n🔍 Scanning for cat/dog images...")
    images = []
    labels = []
    target_size = (64, 64)  
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    files = []

    for f in os.listdir(training_data):
     if f.lower().endswith((".png", ".jpg", ".jpeg")):
        files.append(f)

    total_files = len(files)
    
    for i, filename in enumerate(files):
        try:
            img_path = training_data / filename
            img = imread(img_path)

            if len(img.shape) == 3:  
                img = img.mean(axis=2)  
        
            h, w = img.shape
            img = img[::h//(target_size[0]), ::w//(target_size[1])]
            img = img[:target_size[0], :target_size[1]]
            
            images.append(img.flatten())
            
            if "cat" in filename.lower():
                animal = "🐱"
            else:
                animal="🐶"

            labels.append(0 if animal == "🐱" else 1)

            status_text.text(f"Loaded {i+1}/{total_files}: {animal} image: {filename}")
            progress_bar.progress((i+1)/total_files) #progress() expects a value between 0 and 1-->(100%)
            
        except Exception as e:
            st.warning(f"Skipped {filename} (error: {str(e)})")
            continue

    progress_bar.empty()
    status_text.empty()
    
    return np.array(images), np.array(labels)

def train_model():
    st.write("🧠 Training the AI classifier...")
    x, y = load_images()

    if len(x) == 0:
        st.error("""
        😿 Oops! No valid images found
        Please check if:
        1. The folder path is correct
        2. Images are JPG/PNG format
        3. Filenames contain cat or dog
        """)
    

    st.success(f"✅ Successfully loaded {len(x)} images")
    st.write("⌛ Training the model...")

    model = SVC(kernel="linear", probability=True)
    model.fit(x, y)
    
    st.session_state.model = model
    st.session_state.training_complete = True
    st.session_state.x = x
    st.session_state.y = y
    
    st.success("🎉 Training complete!")
    

def predict_pet(image_path):
    try:
        st.write("🔮 Analyzing image...")
        img = imread(image_path)
        
        if len(img.shape) == 3:
            img = img.mean(axis=2)
            
        h, w = img.shape
        target_size = (64, 64)
        img = img[::h//target_size[0], ::w//target_size[1]]
        img = img[:target_size[0], :target_size[1]]
        
        probabilities = st.session_state.model.predict_proba([img.flatten()])[0]
        cat_prob = probabilities[0]
        dog_prob = probabilities[1]

        prediction = st.session_state.model.predict([img.flatten()])[0]

        if prediction == 0:
            st.success(f"🐱 Result: It's a CAT! (Confidence: {cat_prob*100:.1f}%)")
        else:
            st.success(f"🐕 Result: It's a DOG! (Confidence: {dog_prob*100:.1f}%)")

    except Exception as e:
        st.error(f"❌ Error: Couldn't process image ({str(e)})")


with st.sidebar:
    st.header("Model Training")
    if st.button("Train Model"): #if button got pressed
        train_model()
#(normal layout resumes)


if st.session_state.training_complete:  # if it equals True
    st.header("Image Classification")
    st.write("🎯 Upload an image to classify it as cat or dog")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        
        with open("temp_image", "wb") as f: #open file for binary writing
            f.write(uploaded_file.getbuffer()) #write raw image data
           #closes file handle afterwards
        
 
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
        
        
        predict_pet("temp_image")

        os.remove("temp_image")

else:
    st.info("Please train the model first using the sidebar option.")
