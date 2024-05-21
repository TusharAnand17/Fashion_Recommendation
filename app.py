import streamlit as st
import os
from PIL import Image
import numpy as np
import pickle
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
from numpy.linalg import norm
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm

feature_list = np.array(pickle.load(open('/content/feature_lists.pkl','rb')))
filenames = pickle.load(open('/content/filenames.pkl','rb'))

model = ResNet50(weights = 'imagenet',include_top = False,input_shape = (224,224,3))
model.trainable = False
model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

def feature_extraction(img_path,model):
  img = image.load_img(img_path,target_size = (224,224))
  img_array = image.img_to_array(img)
  expanded_img_array = np.expand_dims(img_array,axis = 0)
  preprocessed_img = preprocess_input(expanded_img_array)
  result = model.predict(preprocessed_img).flatten()
  normalized_result = result / norm(result)
  return normalized_result

def recommend(features,feature_list):
  neighbors = NearestNeighbors(n_neighbors = 5,algorithm='brute',metric = 'euclidean')
  neighbors.fit(feature_list)
  distances,indices = neighbors.kneighbors([features])
  return indices


st.title("Fashion Recommender System")

def save_uploaded_file(uploaded_file):
  try:
    with open(os.path.join('/content/drive/MyDrive/Project fashion Recommender/Uploaded_images',uploaded_file.name),'wb') as f:
      f.write(uploaded_file.getbuffer())
    return 1
  except:
    return 0


uploaded_file = st.file_uploader("Choose an Image: ")
if uploaded_file is not None:
  if save_uploaded_file(uploaded_file):
    display_img = Image.open(uploaded_file)
    st.image(display_img)
    # feature_extraction
    features = feature_extraction(os.path.join('/content/drive/MyDrive/Project fashion Recommender/Uploaded_images',uploaded_file.name),model)
    # st.text(features)
    st.text("Here are the products you may also them...")
    # recommend
    indices = recommend(features,feature_list)
    # display
    col1,col2,col3,col4,col5 = st.columns(5)
    img1 = Image.open(filenames[indices[0][0]])
    col1.image(img1)
    img2 = Image.open(filenames[indices[0][1]])
    col2.image(img2)
    img3 = Image.open(filenames[indices[0][2]])
    col3.image(img3)
    img4 = Image.open(filenames[indices[0][3]])
    col4.image(img4)
    img5 = Image.open(filenames[indices[0][4]])
    col5.image(img5)
  else:
    st.header("Some error occured in file upload")
