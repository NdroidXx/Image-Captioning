from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
import random
import streamlit as st
from PIL import Image

st.title("Image Captioning App Using ViT - By NdroidX")
st.write("Upload an image and generate captions.")

@st.cache_data()

def load_model():
    model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    return model, tokenizer, feature_extractor

model, tokenizer, feature_extractor = load_model()

# Function to generate captions
def generate_captions(image, num_captions):

    # Empty List
    captions = []
    
    # Preprocessing
    pixel_values = feature_extractor(images=[image], return_tensors="pt").pixel_values
    
    # Generating Captions
    for _ in range(num_captions):
        random_seed = random.randint(999, 1000000)
        random.seed(random_seed)
        torch.random.manual_seed(random_seed)

        sampled_output_ids = model.generate(pixel_values, do_sample=True)

        preds = tokenizer.batch_decode(sampled_output_ids, skip_special_tokens=True)
        preds = [pred.strip() for pred in preds]

        # Filter out duplicate captions
        unique_preds = []
        for pred in preds:
            if pred not in unique_preds:
                unique_preds.append(pred)
            if len(unique_preds) == num_captions:
                break

        captions.extend(unique_preds)
    
    return captions

# Image upload and caption generation logic
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Take num_captions as input from the user
num_captions = st.number_input("Enter the number of captions", min_value=1, max_value=10, value=3, step=1)


if uploaded_file is not None:
    image = Image.open(uploaded_file).convert(mode='RGB')
    image = image.resize((500, 400))
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    if st.button("Generate Captions"):
        num_captions = 5  # Number of captions to generate
        captions = generate_captions(image, num_captions)
        st.write("Generated Captions:")
        for caption in captions:
            st.write(caption)
