import streamlit as st
from streamlit_drawable_canvas import st_canvas
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as F

# --- NEW CONSTANT FOR DISPLAY SIZE ---
# This size is used for visualization and the canvas, but NOT for the model input.
DISPLAY_SIZE = 384 

# 1. INITIALIZE
# a) Set Transform
IMG_SIZE = 128 # The size the model expects

# Transform for the MODEL INPUT (128x128)
transform_model = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
])

# --- Model Definition (UNet Architecture) ---
# 
# The U-Net is a seminal architecture for image segmentation and, in this case, inpainting.
# It consists of a contracting path (encoder) to capture context and a symmetric expansive path 
# (decoder) that enables precise localization, connected by skip connections.

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.block(x)

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.down1 = ConvBlock(3, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.down2 = ConvBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.down3 = ConvBlock(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = ConvBlock(256, 512)

        # Decoder
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv3 = ConvBlock(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv2 = ConvBlock(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv1 = ConvBlock(128, 64)

        self.final = nn.Conv2d(64, 3, 1)

    def forward(self, x):
        # Encoder/Downsampling
        d1 = self.down1(x)
        p1 = self.pool1(d1)

        d2 = self.down2(p1)
        p2 = self.pool2(d2)

        d3 = self.down3(p2)
        p3 = self.pool3(d3)

        # Bottleneck
        bn = self.bottleneck(p3)

        # Decoder/Upsampling with Skip Connections (torch.cat)
        u3 = self.up3(bn)
        u3 = torch.cat([u3, d3], dim=1)
        u3 = self.conv3(u3)

        u2 = self.up2(u3)
        u2 = torch.cat([u2, d2], dim=1)
        u2 = self.conv2(u2)

        u1 = self.up1(u2)
        u1 = torch.cat([u1, d1], dim=1)
        u1 = self.conv1(u1)

        return torch.sigmoid(self.final(u1))

# c) Load Model
MODEL_PATH = "inpaint_places256-2.pth" # change the model path here
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Use st.cache_resource for the model to prevent reloading on every interaction
@st.cache_resource
def load_model():
    model = UNet().to(DEVICE)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval()
        return model
    except FileNotFoundError:
        st.error(f"Model file not found at: {MODEL_PATH}. Please check the path.")
        return None

model = load_model()

# 2. SET STREAMLIT APP
st.title("Image Inpainting Model Demo")
st.markdown("Upload an image, draw a mask over the object you want to remove, and click Predict.")

# Sidebar Controls
brush_size = st.sidebar.slider("Brush Size", 5, 50, 15)

# A. File Upload
uploaded_file = st.file_uploader("Upload an image...", type=["png", "jpg", "jpeg"])

if uploaded_file and model:
    # Load and Resize Image (Prepare for Canvas)
    image_pil = Image.open(uploaded_file).convert("RGB")
    
    # *** CHANGE 1: Resize for DISPLAY/CANVAS only ***
    image_display = image_pil.resize((DISPLAY_SIZE, DISPLAY_SIZE))

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Draw Mask Here")
        # B. Interactive Canvas
        # *** CHANGE 2: Set Canvas dimensions to DISPLAY_SIZE ***
        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 0.0)", 
            stroke_width=brush_size,
            stroke_color="#00FF00", # Green brush for visibility
            background_image=image_display, # Use the larger image
            update_streamlit=True,
            height=DISPLAY_SIZE,
            width=DISPLAY_SIZE,
            drawing_mode="freedraw",
            key="canvas",
        )

    # C. Prediction Button
    if st.button("Predict"):
        if canvas_result.image_data is not None:
            with st.spinner('Inpainting...'):
                # --- DATA PROCESSING ---
                
                # 1. Transform Image: Convert to PIL, then resize to MODEL SIZE (128x128)
                # We start with the original PIL image to get the best quality input for the model.
                img_t = transform_model(image_pil).unsqueeze(0).to(DEVICE)
                
                # 2. Process Mask from Canvas
                mask_data = canvas_result.image_data # Shape: (DISPLAY_SIZE, DISPLAY_SIZE, 4)
                alpha_channel = mask_data[:, :, 3] 
                
                # Logic: If pixel has alpha > 0 (it was drawn on), set to 0.0 (remove). Else 1.0 (keep).
                mask_np_full = np.where(alpha_channel > 0, 0.0, 1.0).astype(np.float32)

                # *** NEW STEP: Resize the mask down to the MODEL SIZE (128x128) ***
                mask_pil_full = Image.fromarray((mask_np_full * 255).astype(np.uint8))
                mask_pil_small = mask_pil_full.resize((IMG_SIZE, IMG_SIZE), Image.NEAREST) # Use NEAREST for binary mask
                
                mask_np_small = np.array(mask_pil_small) / 255.0
                mask_t = torch.tensor(mask_np_small).unsqueeze(0).unsqueeze(0).to(DEVICE).float()

                # 3. Create Masked Input (Model size: 128x128)
                masked_img = img_t * mask_t

                # 4. Predict
                with torch.no_grad():
                    pred = model(masked_img)[0].cpu()

                # 5. Combine Prediction with Original
                final_pred = pred * (1 - mask_t.cpu()[0]) + img_t[0].cpu() * mask_t.cpu()[0]

                # --- VISUALIZATION ---
                # Convert back to Numpy (H, W, C) for display (still at 128x128)
                orig_img = img_t[0].cpu().permute(1,2,0).numpy()
                masked_input = masked_img[0].cpu().permute(1,2,0).numpy()
                final_output = final_pred.permute(1,2,0).numpy()

                # Clip values to 0-1 range to prevent display warnings
                final_output = np.clip(final_output, 0, 1)

                # *** NEW STEP: Resize the final images back up to DISPLAY_SIZE for visualization ***
                def resize_for_display(img_array):
                    # Convert float array (0-1) to PIL Image, resize, then back to array
                    img_pil = F.to_pil_image(img_array)
                    img_resized = img_pil.resize((DISPLAY_SIZE, DISPLAY_SIZE), Image.BILINEAR)
                    return np.array(img_resized) / 255.0


                orig_img_disp = resize_for_display(orig_img)
                masked_input_disp = resize_for_display(masked_input)
                final_output_disp = resize_for_display(final_output)

                # Show 3 Columns
                st.write("---")
                st.subheader("Results")
                res_col1, res_col2, res_col3 = st.columns(3)
                
                with res_col1:
                    st.image(orig_img_disp, caption="Original Image", use_column_width=True)
                with res_col2:
                    st.image(masked_input_disp, caption="Masked Input", use_column_width=True)
                with res_col3:
                    st.image(final_output_disp, caption="Predicted Output", use_column_width=True)
        else:
            st.warning("Please draw on the image first!")