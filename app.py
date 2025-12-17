import streamlit as st
from streamlit_drawable_canvas import st_canvas
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from model_archs.unet_arch import UNet
from model_archs.pconv_arch import PConvUNet

# --- CONSTANTS ---
DISPLAY_SIZE = 384 
IMG_SIZE = 128 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

transform_model = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
])

# --- MODEL LOADING LOGIC ---
@st.cache_resource
def load_selected_model(model_name):
    # Map selection to architecture and weights
    if model_name == "Standard UNet":
        model = UNet().to(DEVICE)
        path = "models/unet.pth"
    else:
        model = PConvUNet().to(DEVICE)
        path = "models/pconv_unet2.pth"
    
    try:
        model.load_state_dict(torch.load(path, map_location=DEVICE))
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading {model_name}: {e}")
        return None

# --- STREAMLIT UI ---
st.set_page_config(page_title="Inpainting Showcase", layout="wide")
st.title("Image Inpainting Model Demo")

# Sidebar
st.sidebar.header("Model Configuration")
model_choice = st.sidebar.selectbox("Select Architecture", ["Standard UNet", "PConv UNet"])
brush_size = st.sidebar.slider("Brush Size", 5, 50, 20)
st.sidebar.info(f"Currently using: {model_choice}")

# Load model based on selection
model = load_selected_model(model_choice)

uploaded_file = st.file_uploader("Upload an image to start...", type=["png", "jpg", "jpeg"])

if uploaded_file and model:
    image_pil = Image.open(uploaded_file).convert("RGB")
    image_display = image_pil.resize((DISPLAY_SIZE, DISPLAY_SIZE))

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Input & Masking")
        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 0.0)", 
            stroke_width=brush_size,
            stroke_color="#00FF00", 
            background_image=image_display,
            update_streamlit=True,
            height=DISPLAY_SIZE,
            width=DISPLAY_SIZE,
            drawing_mode="freedraw",
            key="canvas",
        )
        st.caption("Draw over the area you want the model to 'fill in'.")

    if st.button("Run Inpainting"):
        if canvas_result.image_data is not None and np.any(canvas_result.image_data[:, :, 3] > 0):
            with st.spinner(f'Processing with {model_choice}...'):
                # 1. Prepare Image Tensor
                img_t = transform_model(image_pil).unsqueeze(0).to(DEVICE)
                
                # 2. Process Mask Tensor
                mask_data = canvas_result.image_data 
                alpha_channel = mask_data[:, :, 3] 
                mask_np_full = np.where(alpha_channel > 0, 0.0, 1.0).astype(np.float32)
                mask_pil_full = Image.fromarray((mask_np_full * 255).astype(np.uint8))
                mask_pil_small = mask_pil_full.resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)
                mask_np_small = np.array(mask_pil_small) / 255.0
                mask_t = torch.tensor(mask_np_small).unsqueeze(0).unsqueeze(0).to(DEVICE).float()

                # 3. Create Masked Input (Visual feedback)

                # 4. Predict
                with torch.no_grad():
                    masked_img = img_t * mask_t

                    if model_choice == "PConv UNet":
                        pred = model(masked_img, mask_t)[0].cpu()
                    else:
                        pred = model(masked_img)[0].cpu()

                # 5. Post-processing: Blend prediction into original image holes
                final_pred = pred * (1 - mask_t.cpu()[0]) + img_t[0].cpu() * mask_t.cpu()[0]
                
                def tensor_to_disp(tensor):
                    return TF.to_pil_image(torch.clamp(tensor, 0, 1)).resize((DISPLAY_SIZE, DISPLAY_SIZE))

                # Display Results
                st.divider()
                st.subheader("Results")
                res_col1, res_col2, res_col3 = st.columns(3)
                
                res_col1.image(tensor_to_disp(img_t[0].cpu()), caption="Original (Downscaled)")
                res_col2.image(tensor_to_disp(masked_img[0].cpu()), caption="Model Input")
                res_col3.image(tensor_to_disp(final_pred), caption=f"Result ({model_choice})")
        else:
            st.warning("Please draw a mask on the canvas first!")