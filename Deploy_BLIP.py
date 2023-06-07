from transformers import BlipForConditionalGeneration, BlipProcessor, AutoTokenizer
import torch
import random
import streamlit as st
from PIL import Image

st.title("Image Captioning App Using BLIP - By NdroidX")
st.write("Upload an image and generate captions.")

@st.cache_data()

def load_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
    tokenizer = AutoTokenizer.from_pretrained("Salesforce/blip-image-captioning-large")
    return processor, model, tokenizer

processor, model, tokenizer = load_model()

# Function to generate captions
def generate_captions(image, num_captions):
    # Empty list to store captions
    captions = []

    inputs = processor(image, return_tensors="pt")

    # Generate multiple captions
    while len(captions) < num_captions:
        random_seed = random.randint(999, 1000000)
        random.seed(random_seed)
        torch.manual_seed(random_seed)

        out = model.generate(
            **inputs,
            num_return_sequences=num_captions,
            do_sample=True,
            top_k=100,
            temperature=0.7,
            max_length=50
        )
        
        for i in range(num_captions):
            caption = processor.decode(out[i], skip_special_tokens=True)
            captions.append(caption)
    
    return captions



# Image upload and caption generation logic
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Take num_captions as input from the user
num_captions = st.number_input("Enter the number of captions", min_value=1, max_value=10, value=3, step=1)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    if st.button("Generate Captions"):
        captions = generate_captions(image, num_captions)
        st.write("Generated Captions:")
        for caption in captions:
            st.write(caption.capitalize())
