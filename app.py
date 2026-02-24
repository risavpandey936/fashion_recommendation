import streamlit as st
import os
import pickle
import numpy as np
import pandas as pd
from PIL import Image
import zipfile
import gdown
import shutil

import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input

from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm

# ==================================================
# Streamlit Config
# ==================================================
st.set_page_config(page_title="Fashion Recommender", layout="wide")
st.title("👕 Fashion Recommendation System")

# ==================================================
# DATA CONFIG
# ==================================================
DATA_DIR = "data"
IMAGE_DIR = os.path.join(DATA_DIR, "images")
UPLOAD_DIR = "uploads"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

FILES = {
    "images.zip": "https://drive.google.com/uc?id=1GS3Ocqs_wfbeFQGaDM5qzFYh72p6xSQS",
    "filenames.pkl": "https://drive.google.com/uc?id=1LSqYhMrNmT_N5gwXlQ8w8TumFWXDQZBX",
    "embeddings.pkl": "https://drive.google.com/uc?id=1C9_dKaXq1kwSRB4cW7pygdSjgmWWZxfx"
}

def download_if_missing():
    for file, url in FILES.items():
        path = os.path.join(DATA_DIR, file)
        if not os.path.exists(path):
            st.info(f"Downloading {file}...")
            gdown.download(url, path, quiet=False)

    # Extract images safely
    if not os.path.exists(IMAGE_DIR):
        st.info("Extracting images...")
        with zipfile.ZipFile(os.path.join(DATA_DIR, "images.zip"), "r") as z:
            z.extractall(DATA_DIR)

        # Handle nested images/images case
        nested = os.path.join(IMAGE_DIR, "images")
        if os.path.exists(nested):
            for f in os.listdir(nested):
                shutil.move(os.path.join(nested, f), IMAGE_DIR)
            shutil.rmtree(nested)

download_if_missing()

# ==================================================
# Load Embeddings
# ==================================================
@st.cache_resource
def load_embeddings():
    features = np.array(
        pickle.load(open(os.path.join(DATA_DIR, "embeddings.pkl"), "rb"))
    )
    filenames = pickle.load(
        open(os.path.join(DATA_DIR, "filenames.pkl"), "rb")
    )
    return features, filenames

feature_list, filenames = load_embeddings()

# ==================================================
# Load Metadata
# ==================================================
@st.cache_data
def load_metadata():
    df = pd.read_csv("styles.csv", on_bad_lines="skip", encoding="utf-8")
    df["id"] = df["id"].astype(str)
    return df

metadata = load_metadata()

# ==================================================
# Load CNN Model
# ==================================================
@st.cache_resource
def load_model():
    base = ResNet50(weights="imagenet", include_top=False, input_shape=(224,224,3))
    base.trainable = False
    return tf.keras.Sequential([base, GlobalMaxPooling2D()])

model = load_model()

# ==================================================
# Build KNN
# ==================================================
@st.cache_resource
def build_knn(features):
    knn = NearestNeighbors(n_neighbors=6, metric="euclidean", algorithm="brute")
    knn.fit(features)
    return knn

knn_model = build_knn(feature_list)

# ==================================================
# Feature Extraction
# ==================================================
def extract_features(img_path):
    img = image.load_img(img_path, target_size=(224,224))
    arr = image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    vec = model.predict(arr, verbose=0).flatten()
    return vec / norm(vec)

# ==================================================
# Recommendation Logic
# ==================================================
def recommend(vec):
    _, idx = knn_model.kneighbors([vec])
    return idx[0][1:]

def get_product_id(path):
    return os.path.splitext(os.path.basename(path))[0]

# ==================================================
# UI
# ==================================================
uploaded = st.file_uploader("Upload a fashion image", type=["jpg","jpeg","png"])

if uploaded:
    path = os.path.join(UPLOAD_DIR, uploaded.name)
    with open(path, "wb") as f:
        f.write(uploaded.getbuffer())

    st.image(Image.open(path), width=250)
    vec = extract_features(path)
    indices = recommend(vec)

    st.subheader("🛍️ Recommended Fashion Items")
    cols = st.columns(5)

    for col, i in zip(cols, indices):
        img_name = os.path.basename(filenames[i])
        img_path = os.path.join(IMAGE_DIR, img_name)
        info = metadata[metadata["id"] == get_product_id(img_name)]

        with col:
            if os.path.exists(img_path):
                st.image(img_path, use_container_width=True)
            else:
                st.caption("Image missing")

            if not info.empty:
                info = info.iloc[0]
                st.markdown(f"""
                **Category:** {info['masterCategory']}  
                **Sub-category:** {info['subCategory']}  
                **Type:** {info['articleType']}  
                **Gender:** {info['gender']}  
                **Color:** {info['baseColour']}  
                **Season:** {info['season']}  
                **Year:** {info['year']}
                """)