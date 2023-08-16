import streamlit as st
from PIL import Image
import numpy as np
import requests

# Function to send features to an API for localization prediction
def decode(features):
    # Create the request body
    body = {"features":features}
    try:
        # Send a POST request to the localization prediction API
        response = requests.post(url="http://localhost:105/sliceLocalizationPrediction", json=body)
    except requests.exceptions.ConnectionError as e:
         # Handle connection error to the API
        st.text("API Connection Failed")
        return []

    if response.status_code == 200:
        imageResponse = np.array(response.json()['image'])
        return imageResponse
    else:
        # Handle API call failure
        st.text("API Call Failed")

def main():
     # Streamlit app title and user input section
    st.title("CATSCAN Localization")
    featuresInput = st.text_input("Enter comma separted value")
    
    # Button to trigger localization prediction
    if st.button("Localize"):
        features = list(map(float,featuresInput.title().split(",")))
        img = decode(features) # Call the localization prediction function

        if len(img)>0:
            # Display the predicted image
            st.image(img, width=391)

if __name__ == "__main__":
    main() # Run the Streamlit app
