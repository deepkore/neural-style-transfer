import streamlit as st
import torch
from torchvision import transforms
from transformer_net import TransformerNet 
from PIL import Image
import os

# Function to load the selected model

# Streamlit UI
st.title("Fast Neural Style Transfer")

# Get available models from 'saved_models/' directory
model_dir = "saved_models"
image_dir = "style_images"  # Folder where style images are stored

model_files = [f for f in os.listdir(model_dir) if f.endswith(".pth")]
model_names = [os.path.splitext(f)[0] for f in model_files]
image_files = {name: os.path.join(image_dir, f"{name}.jpg") for name in model_names if os.path.exists(os.path.join(image_dir, f"{name}.jpg"))}


if not model_files:
    st.error("No models found in 'saved_models/' folder. Please add .pth models.")
else:
    # Allow user to select a style model
       # Sidebar Image Selector (linked to models)
    selected_style = st.sidebar.selectbox("Choose a Style:", model_names)

    # Display corresponding style image
    if selected_style in image_files:
        st.sidebar.image(image_files[selected_style], caption=f"Style: {selected_style}", width=150)

    # Load the corresponding model
    model_path = os.path.join(model_dir, f"{selected_style}.pth")
    state_dict = torch.load(model_path)

    # Remove running_mean and running_var from InstanceNorm layers
    for key in list(state_dict.keys()):
        if "running_mean" in key or "running_var" in key:
            del state_dict[key]

    model = TransformerNet()
    model.load_state_dict(state_dict, strict=False)  # Use strict=False to ignore extra keys
    model.eval()

    # Upload an image
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Original Image", width=300)
        transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.mul(255))
        ])

        image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

        with torch.no_grad():
            output_tensor = model(image_tensor).squeeze(0).cpu()

        output_image = transforms.ToPILImage()(output_tensor / 255)
        st.image(output_image, caption="Stylized Image", width=300)
