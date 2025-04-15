
import os
import numpy as np
from sklearn.svm import SVC #machine learning
from matplotlib.image import imread #image processing
from pathlib import Path
from PIL import Image


print(r"""      
   /\_/\           /)__)
  ( o.o )         (='.'=)    
   > ^ <          (")_(")   
""")
print("🌟 Welcome to the Interactive Cat/Dog Classifier! 🌟\n")


training_data = Path("pet_photos")
 


def load_images():
    print("\n🔍 Scanning for cat/dog images...")
    images = []
    labels = []
    target_size = (64, 64)  
    
    #lists the files
    for filename in os.listdir(training_data):
        if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
            continue 

    #(2, 2, 3) → (height, width, channels) color
    #(2, 2) → (image height, image width) grayscale (2 rows tall 2 columns wide)

        try:
            img_path = training_data / filename
            img = imread(img_path)

    #axis=0: operates vertically (along rows)
    #axis=1: operates horizontally (along columns)
    #axis=2: operates across color channels (for colored images)
            
            if len(img.shape) == 3:  
                img = img.mean(axis=2)  
        
            h, w = img.shape

    #resizing to match target size
    #devision calculates how many pixels to skip to reach target size
    #height: vertical skip (row by row)
    # width: horizontal skip (column by column)

            #start/stop/step 
            img = img[::h//(target_size[0]), ::w//(target_size[1])]
            img = img[:target_size[0], :target_size[1]] #safety crop
            
            images.append(img.flatten()) #2d to 1d (need to flatten for SVM input)
            
       
            if "cat" in filename.lower():
                animal = "🐱"
            else:
                animal="🐶"
            labels.append(0 if animal == "🐱" else 1)

            
            print(f"  Loaded {animal} image: {filename}")
            
        except Exception as e:
            print(f"  ❌ Skipped {filename} (error: {str(e)})")
            continue

    #each image's flattened pixels become one row in a 2D array
    return np.array(images), np.array(labels)
 

print("\n🧠 Training the AI classifier...")
x, y = load_images()

if len(x) == 0:
    print("\n😿 Oops! No valid images found")
    print("   Please check if:")
    print("   1. The folder path is correct")
    print("   2. Images are JPG/PNG format")
    print("   3. Filenames contain cat or dog")
    exit()

print(f"\n✅ Successfully loaded {len(x)} images!")
print("   ⌛ Training the model...")


#decision boundary is linear
model = SVC(kernel="linear", probability=True) #enables confidence estimates
model.fit(x, y)


print("   🎉 Training complete!")


def predict_pet(image_path):
    try:
        print(f"\n🔮 Analyzing image...")
        img = imread(image_path.strip('"'))
        
    #process exactly like training images
        if len(img.shape) == 3:
            img = img.mean(axis=2)
            
        h, w = img.shape
        target_size = (64, 64)
        img = img[::h//target_size[0], ::w//target_size[1]]
        img = img[:target_size[0], :target_size[1]]
        

    #model.predict_proba and model.predict expect multiple samples even if you only have one
    # ^^ which makes adding the [] around img.flatten necessary in order to create a batch
    # [0] removes the extra batch at the end

        probabilities = model.predict_proba([img.flatten()])[0]
        cat_prob = probabilities[0]  #probabilities= [catprob,dogprob]
        dog_prob = probabilities[1]
 

        prediction = model.predict([img.flatten()])[0]


        if prediction == 0:
            return (f"   🐱 Result: It's a CAT! (Confidence: {cat_prob*100:.1f}%)")
        else:
            return (f"   🐕 Result: It's a DOG! (Confidence: {dog_prob*100:.1f}%)")

    except Exception as e:
        return f"   ❌ Error: Couldn't process image ({str(e)})"



print("\n🎯 STEP 2: Let's classify an image!")
while True:
    print("\n👉 Please enter the full path to an image to classify (or type exit):")
    print("   Example: C:\\Users\\You\\Desktop\\test_pet.jpg")


    user_input = input("> ").strip('"')
    
    if user_input.lower() == "exit":
        print("\n✨ Thanks for using the Cat/Dog Classifier! Goodbye! ✨")
        break
        
    if not os.path.exists(user_input):
        print("   ❌ File not found! Please try again.")
        continue

        
    print(predict_pet(user_input))
    

    print("\nWould you like to classify another image? (yes/no)")
    if input("> ").lower() != 'yes':
        print("\n✨ Thanks for using the Cat/Dog Classifier! Goodbye! ✨")
        break
