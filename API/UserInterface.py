import streamlit as st
from PIL import Image
import numpy as np
import requests

def decode(features):
    body = {"features":features}
    try:
        response = requests.post(url="http://localhost:105/sliceLocalizationPrediction", json=body)
    except requests.exceptions.ConnectionError as e:
        st.text("API Connection Failed")
        return []

    if response.status_code == 200:
        imageResponse = np.array(response.json()['image'])
        return imageResponse
    else:
        st.text("API Call Failed")

def main():
    st.title("CATSCAN Localization")
    featuresInput = st.text_input("Enter comma separted value")
    if st.button("Localize"):
        features = list(map(float,featuresInput.title().split(",")))
        img = decode(features)
        if len(img)>0:
            st.image(img, width=391)

if __name__ == "__main__":
    main()