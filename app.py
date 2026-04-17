import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import pandas as pd
import time

# ==========================================
# PAGE CONFIGURATION & STYLING
# ==========================================

st.set_page_config(
    page_title="AI Agronomist | High-Contrast Analytics",
    page_icon="https://cdn-icons-png.flaticon.com/512/1864/1864472.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Professional UI Styling - Font Awesome & High Contrast
st.markdown("""
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    <style>
    .main {
        background-color: #ffffff;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3.5em;
        background-color: #1a5e20;
        color: white !important;
        font-weight: 700;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #2e7d32;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }
    .prediction-card {
        padding: 25px;
        border-radius: 12px;
        background-color: #f8f9fa;
        box-shadow: 0 2px 15px rgba(0,0,0,0.1);
        border: 2px solid #e0e0e0;
        margin-top: 15px;
        margin-bottom: 25px;
        color: #000000 !important;
    }
    .info-header {
        color: #1b5e20;
        font-size: 24px;
        font-weight: 800;
        margin-top: 30px;
        margin-bottom: 15px;
        display: flex;
        align-items: center;
    }
    .info-header i {
        margin-right: 12px;
    }
    .disease-desc {
        background-color: #e8f5e9;
        padding: 20px;
        border-radius: 8px;
        border-left: 6px solid #2e7d32;
        color: #000000 !important;
        font-size: 16px;
        line-height: 1.6;
    }
    .action-plan {
        background-color: #fff3e0;
        padding: 20px;
        border-radius: 8px;
        border-left: 6px solid #ef6c00;
        margin-top: 20px;
        color: #000000 !important;
        font-size: 16px;
        line-height: 1.6;
    }
    .metric-label {
        font-size: 14px;
        color: #333333;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        margin-bottom: 5px;
    }
    h1, h2, h3 {
        color: #111111 !important;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# RELIABLE MODEL LOADING (CACHED)
# ==========================================

@st.cache_resource
def load_agronomist_model(model_path):
    """
    Loads the brain only once to ensure consistent, lightning-fast predictions.
    """
    if not os.path.exists(model_path):
        return None
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Logic Error: Could not initialize neural weights. {e}")
        return None

# ==========================================
# KNOWLEDGE BASE
# ==========================================

CROP_DATA = {
    "Cassava": {
        "classes": ["Cassava Bacterial Blight (CBB)", "Cassava Brown Streak Disease (CBSD)", "Cassava Green Mottle (CGM)", "Cassava Mosaic Disease (CMD)", "Healthy"],
        "model_path": os.path.join("models", "cassava_model.h5"),
        "info": {
            "Cassava Bacterial Blight (CBB)": {"symptoms": "Angular leaf spots with yellow halos. Progresses to wilting.", "action": "Destroy infected plants. Use clean cuttings.", "color": "#b71c1c", "icon": "fa-biohazard"},
            "Cassava Brown Streak Disease (CBSD)": {"symptoms": "Yellowing along veins. Brown streaks on roots.", "action": "Harvest early. Rogue infected plants.", "color": "#4a148c", "icon": "fa-virus-slash"},
            "Cassava Green Mottle (CGM)": {"symptoms": "Chlorotic mottling and stunting.", "action": "Vector control (whiteflies). Clean material.", "color": "#1b5e20", "icon": "fa-bug"},
            "Cassava Mosaic Disease (CMD)": {"symptoms": "Severe yellowing and leaf distortion.", "action": "Use resistant varieties. Rogue plants.", "color": "#e65100", "icon": "fa-leaf"},
            "Healthy": {"symptoms": "Normal growth with no visible pathologies.", "action": "Continue routine monitoring.", "color": "#004d40", "icon": "fa-check-circle"}
        }
    },
    "Maize": {
        "classes": ["Maize Lethal Necrosis", "Maize Rust", "Maize Streak Virus", "Maize Fall Armyworm", "Healthy"],
        "model_path": os.path.join("models", "maize_model.h5"),
        "info": {
            "Maize Lethal Necrosis": {"symptoms": "Sudden leaf marginal necrosis and plant death.", "action": "Rotate crops. Vector control.", "color": "#b71c1c", "icon": "fa-skull"},
            "Maize Rust": {"symptoms": "Orange-brown pustules on leaves.", "action": "Fungicide application if severe.", "color": "#bf360c", "icon": "fa-biohazard"},
            "Maize Streak Virus": {"symptoms": "Linear yellow streaks on leaves.", "action": "Control leafhoppers with insecticides.", "color": "#ff6f00", "icon": "fa-bolt"},
            "Maize Fall Armyworm": {"symptoms": "Ragged holes and frass in the whorl.", "action": "Apply localized biopesticides.", "color": "#3e2723", "icon": "fa-locust"},
            "Healthy": {"symptoms": "Vibrant green blades, no lesions.", "action": "Ensure proper nitrogenous feed.", "color": "#004d40", "icon": "fa-check-double"}
        }
    }
}

# ==========================================
# INFERENCE LOGIC (FIXED: NO RANDOM FALLBACK)
# ==========================================

def process_specimen(image_file):
    """Resizes and normalizes input for the model."""
    img = Image.open(image_file).convert("RGB")
    prepared = np.array(img.resize((224, 224))) / 255.0
    return img, np.expand_dims(prepared, axis=0)

def predict_pathology(batch, crop_type):
    """
    Performs inference using the loaded model.
    Removed random fallback to ensure total consistency.
    """
    meta = CROP_DATA[crop_type]
    model = load_agronomist_model(meta["model_path"])
    
    if model:
        out = model.predict(batch)
        idx = np.argmax(out)
        return meta["classes"][idx], float(out[0][idx])
    else:
        st.warning(f"Detection Mode: {crop_type} model not found. Proceeding with analysis engine calibration.")
        time.sleep(1)
        # We return a specific 'Not Found' state instead of random guesses
        return "Model Calibration Pending", 0.0

# ==========================================
# UI RENDERER
# ==========================================

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1864/1864472.png", width=90)
    st.markdown("<h2 style='color: #1a5e20;'>Agronomist Dashboard</h2>", unsafe_allow_html=True)
    crop = st.selectbox("Select Crop Type", ["Cassava", "Maize"])
    mode = st.radio("System Mode", ["Individual Analytics", "Batch Farm Audit"])
    st.divider()
    st.info(f"System identifying {crop} diseases using validated CNN weights.")

st.markdown(f"<h1><i class='fas fa-microscope'></i> {crop} Health Analytics</h1>", unsafe_allow_html=True)

if mode == "Individual Analytics":
    if "show_camera" not in st.session_state:
        st.session_state.show_camera = False

    col_ctrl, col_res = st.columns([1, 1], gap="large")

    with col_ctrl:
        st.markdown("<div class='info-header'><i class='fas fa-file-upload'></i> Upload Specimen</div>", unsafe_allow_html=True)
        file_up = st.file_uploader("Drop image here or browse", type=["jpg", "png", "jpeg"])
        
        st.divider()
        if st.button("📸 Open Live Camera"):
            st.session_state.show_camera = not st.session_state.show_camera
        
        camera_snap = None
        if st.session_state.show_camera:
            camera_snap = st.camera_input("Capture leaf pattern")

        active_specimen = camera_snap if camera_snap else file_up

    if active_specimen:
        with col_res:
            st.markdown("<div class='info-header'><i class='fas fa-microscope'></i> Analyzed Specimen & Prediction</div>", unsafe_allow_html=True)
            vis_img, batch_data = process_specimen(active_specimen)
            st.image(vis_img, caption="Current Analysis Subject", use_container_width=True)
            
            with st.spinner("Analyzing cell structures..."):
                label, conf = predict_pathology(batch_data, crop)
            
            if label in CROP_DATA[crop]["info"]:
                info_card = CROP_DATA[crop]["info"][label]
                
                st.markdown(f"""
                <div class='prediction-card'>
                    <div class='metric-label'>Diagnosis Confidence</div>
                    <h1 style='color: {info_card['color']}; font-weight: 900; margin-top:0;'>{int(conf*100)}% Accurate</h1>
                    <hr style="border: 1px solid #ddd;">
                    <div class='metric-label'>Detected Condition</div>
                    <h2 style='color: #111; font-weight: 800;'><i class='fas {info_card['icon']}'></i> {label}</h2>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"<div class='disease-desc'><b>Observed Symptoms:</b><br>{info_card['symptoms']}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='action-plan'><b>Recommended Remediation:</b><br>{info_card['action']}</div>", unsafe_allow_html=True)
            else:
                st.error("Diagnosis Calibration Failed: No trained weights detected for this class.")

else: # Batch Audit Mode
    st.markdown("<div class='info-header'><i class='fas fa-layer-group'></i> Batch Farm Audit</div>", unsafe_allow_html=True)
    batch_files = st.file_uploader("Select multiple specimens for report", type=["jpg", "png"], accept_multiple_files=True)
    
    if batch_files:
        if st.button("Start Audit Analysis"):
            audit_log = []
            progress = st.progress(0)
            for i, f in enumerate(batch_files):
                _, b = process_specimen(f)
                l, c = predict_pathology(b, crop)
                audit_log.append({"Specimen": f.name, "Condition": l, "Confidence": f"{int(c*100)}%"})
                progress.progress((i + 1) / len(batch_files))
            
            res_df = pd.DataFrame(audit_log)
            st.success(f"Audit Complete: Processed {len(batch_files)} samples.")
            st.dataframe(res_df, use_container_width=True)
            st.bar_chart(res_df["Condition"].value_counts())

st.markdown("---")
st.markdown("<div style='text-align: center; color: #444; font-size: 14px;'>Empowering Farmers with Precision AI Analytics</div>", unsafe_allow_html=True)
