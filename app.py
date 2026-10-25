from pathlib import Path
from io import BytesIO

import numpy as np
import onnxruntime as ort
import streamlit as st
from PIL import Image, ImageOps

MODEL_PATH = Path(__file__).parent / "unet_baseline.onnx"

IMAGE_SIZE = 256
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    return ort.InferenceSession(
        str(MODEL_PATH),
        providers=["CPUExecutionProvider"]
    )


def preprocess_image(image):
    image = image.convert("RGB")
    resized = image.resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        Image.Resampling.BILINEAR
    )

    array = np.asarray(resized, dtype=np.float32) / 255.0
    array = (array - MEAN) / STD

    # HWC -> CHW -> NCHW
    array = np.transpose(array, (2, 0, 1))
    array = np.expand_dims(array, axis=0)

    return np.ascontiguousarray(array, dtype=np.float32)


def predict_mask(image, session, threshold):
    input_array = preprocess_image(image)

    logits = session.run(
        ["output"],
        {"input": input_array}
    )[0]

    # Convert logits to probabilities
    probabilities = 1.0 / (
        1.0 + np.exp(-np.clip(logits, -50, 50))
    )

    mask = (probabilities[0, 0] > threshold).astype(np.uint8)
    mask_image = Image.fromarray(mask * 255, mode="L")

    # Restore original image size
    mask_image = mask_image.resize(
        image.size,
        Image.Resampling.NEAREST
    )

    return mask_image


def create_overlay(image, mask):
    original = image.convert("RGB")
    original_array = np.asarray(original, dtype=np.float32)
    mask_array = np.asarray(mask) > 0

    overlay_array = original_array.copy()
    red = np.array([255, 60, 80], dtype=np.float32)

    overlay_array[mask_array] = (
        0.55 * original_array[mask_array]
        + 0.45 * red
    )

    return Image.fromarray(
        np.clip(overlay_array, 0, 255).astype(np.uint8)
    )


def create_transparent_image(image, mask):
    result = image.convert("RGBA")
    result.putalpha(mask)
    return result


def image_to_png_bytes(image):
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


st.set_page_config(
    page_title="PetVision Studio",
    page_icon="🐾",
    layout="wide"
)

st.title("PetVision Studio")
st.write(
    "AI-powered pet segmentation using a custom U-Net model."
)

st.caption(
    "Upload a cat or dog image to separate the pet "
    "from its background."
)

threshold = st.sidebar.slider(
    "Segmentation threshold",
    min_value=0.1,
    max_value=0.9,
    value=0.5,
    step=0.05
)

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        image = ImageOps.exif_transpose(image).convert("RGB")

        session = load_model()

        with st.spinner("Segmenting pet..."):
            mask = predict_mask(image, session, threshold)
            overlay = create_overlay(image, mask)
            transparent = create_transparent_image(image, mask)

        st.success("Segmentation complete!")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Original")
            st.image(image, use_container_width=True)

        with col2:
            st.subheader("Predicted Mask")
            st.image(mask, use_container_width=True)

        with col3:
            st.subheader("Overlay")
            st.image(overlay, use_container_width=True)

        st.subheader("Transparent Background")

        st.image(
            transparent,
            use_container_width=True
        )

        st.download_button(
            label="Download transparent PNG",
            data=image_to_png_bytes(transparent),
            file_name="petvision_segmentation.png",
            mime="image/png"
        )

    except Exception as error:
        st.error(f"Could not process image: {error}")

st.divider()

st.caption(
    "Model: Custom U-Net | "
    "Dataset: Oxford-IIIT Pet | "
    "Inference: ONNX Runtime"
)
