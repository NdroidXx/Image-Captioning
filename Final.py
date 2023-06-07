from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from transformers import BlipForConditionalGeneration, BlipProcessor, AutoTokenizer
import torch
import random
import streamlit as st
from PIL import Image

st.title("Image Captioning App - By NdroidX")
st.write("Upload an image and generate captions.")

@st.cache_data()

def load_model_vit():
    model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    return model, tokenizer, feature_extractor

model_vit, tokenizer_vit, feature_extractor_vit = load_model_vit()

def load_model_blip():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
    tokenizer = AutoTokenizer.from_pretrained("Salesforce/blip-image-captioning-large")
    return processor, model, tokenizer

processor_blip, model_blip, tokenizer_blip = load_model_blip()


# Function to generate captions using ViT
def generate_captions_vit(image, num_captions):

    # Empty List
    captions = []
    
    # Preprocessing
    pixel_values = feature_extractor_vit(images=[image], return_tensors="pt").pixel_values
    
    # Generating Captions
    for _ in range(num_captions):
        random_seed = random.randint(999, 1000000)
        random.seed(random_seed)
        torch.random.manual_seed(random_seed)

        sampled_output_ids = model_vit.generate(pixel_values, do_sample=True)

        preds = tokenizer_vit.batch_decode(sampled_output_ids, skip_special_tokens=True)
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

# Function to generate captions using BLIP
def generate_captions_blip(image, num_captions):
    # Empty list to store captions
    captions = []

    inputs = processor_blip(image, return_tensors="pt")

    # Generate multiple captions
    while len(captions) < num_captions:
        random_seed = random.randint(999, 1000000)
        random.seed(random_seed)
        torch.manual_seed(random_seed)

        out = model_blip.generate(
            **inputs,
            num_return_sequences=num_captions,
            do_sample=True,
            top_k=100,
            temperature=0.7,
            max_length=50
        )
        
        for i in range(num_captions):
            caption = processor_blip.decode(out[i], skip_special_tokens=True)
            captions.append(caption)
    
    return captions

# Image upload and caption generation logic
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Take num_captions as input from the user
num_captions = st.number_input("Enter the number of captions", min_value=1, max_value=10, value=3, step=1)


if uploaded_file is not None:
    image = Image.open(uploaded_file).convert(mode='RGB')
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    if st.button("Generate Captions"):
        captions_vit = generate_captions_vit(image, num_captions)
        captions_blip = generate_captions_blip(image, num_captions)

        st.markdown("\t Generated Captions Using ViT + GPT2 : ")
        for caption in captions_vit:
            st.write(caption.capitalize())
        st.markdown("\t Generated Captions Using BLIP : ")
        for caption in captions_blip:
            st.write(caption.capitalize())

