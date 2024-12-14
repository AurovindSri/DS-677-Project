import streamlit as st
import requests
from PIL import Image
import io

# Title of the app
st.title('CIFAR-10 Image Prediction')

# Instructions for users
st.write("Upload an image to classify it into one of the CIFAR-10 categories:")

# Upload image input
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Open the image using PIL
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Convert image to bytes for FastAPI
    image_bytes = uploaded_file.getvalue()

    # FastAPI backend URL (Docker container's IP address or localhost if running locally)
    #backend_url = "http://localhost:8085/predict"  # Update this if running in Docker
    backend_url = "http://localhost:8085/predict"

    # Send the image to the FastAPI server
    files = {'file': ('image.jpg', image_bytes, 'image/jpeg')}
    response = requests.post(backend_url, files=files)

    # Check if prediction is successful and display result
    if response.status_code == 200:
        #print("inside if condition")
        prediction = response.json()
        #st.write(prediction)
        st.write(f"Predicted Class: {prediction['class_name']}")
        st.write(f"Probability: {prediction['probability']:.2f}")
    else:
        st.write("Error with the prediction request.")
        #docker tag local-image:tagname new-repo:tagname
#docker push new-repo:tagname
