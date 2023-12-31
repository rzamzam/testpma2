import streamlit as st
from keras.models import load_model
from PIL import Image
import numpy as np

from util import classify, set_background


set_background('./BG/bg.jpg')

# set title
st.title('Pneumonia covid classification')

# set header
st.header('Please upload a chest X-ray image')

# upload file
file = st.file_uploader('', type=['jpeg', 'jpg', 'png'])

# load classifier
model_path = './model/fixedmodel.h5' 
loaded_model = load_model(model_path, compile=False)

# Define your class names
class_names = {0: 'Normal', 1: 'Viral Pneumonia', 2: 'Covid'}

# load class names
# with open('./model/labels.txt', 'r') as f:
#     class_names = [a[:-1].split(' ')[1] for a in f.readlines()]
#     f.close()

try:
    image = Image.open(image_path).convert('RGB')
except Exception as e:
    print(f"Error opening image: {image}")
    print(f"Exception details: {e}")
    
# display image
if file is not None:
    image = Image.open(file).convert('RGB')
    st.image(image, use_column_width=True)

    # classify image
    class_name, conf_score = classify(image, loaded_model, class_names)

    # write classification
    st.write("## {}".format(class_name))
    st.write("### score: {}%".format(int(conf_score * 1000) / 10))
