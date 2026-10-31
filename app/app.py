import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import pickle
from PIL import Image
import os
import pandas as pd
import io

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="PneumoScan AI | Clinical Diagnostic Suite",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Vector SVG Icons Library (No Emojis)
# ---------------------------------------------------------
def get_svg_icon(name, color="currentColor", size=18):
    icons = {
        "lungs": f'''<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; display: inline-block;"><path d="M6.5 17.5C4 16 3 13 3 10a7 7 0 0 1 7-7v12.5"/><path d="M17.5 17.5C20 16 21 13 21 10a7 7 0 0 0-7-7v12.5"/><path d="M12 3v18"/><path d="M12 10l-4 4"/><path d="M12 10l4 4"/><circle cx="12" cy="21" r="1"/></svg>''',
        "check": f'''<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; display: inline-block;"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>''',
        "alert": f'''<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; display: inline-block;"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>''',
        "activity": f'''<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; display: inline-block;"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>''',
        "cpu": f'''<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; display: inline-block;"><rect x="4" y="4" width="16" height="16" rx="2" ry="2"/><rect x="9" y="9" width="6" height="6"/><line x1="9" y1="1" x2="9" y2="4"/><line x1="15" y1="1" x2="15" y2="4"/><line x1="9" y1="20" x2="9" y2="23"/><line x1="15" y1="20" x2="15" y2="23"/><line x1="20" y1="9" x2="23" y2="9"/><line x1="20" y1="14" x2="23" y2="14"/><line x1="1" y1="9" x2="4" y2="9"/><line x1="1" y1="14" x2="4" y2="14"/></svg>''',
        "shield": f'''<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; display: inline-block;"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>''',
        "shield_check": f'''<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; display: inline-block;"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><polyline points="9 12 11 14 15 10"/></svg>''',
        "shield_alert": f'''<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; display: inline-block;"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>''',
        "chart": f'''<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; display: inline-block;"><line x1="12" y1="20" x2="12" y2="10"/><line x1="18" y1="20" x2="18" y2="4"/><line x1="6" y1="20" x2="6" y2="16"/></svg>''',
        "file": f'''<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; display: inline-block;"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/></svg>''',
        "layers": f'''<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; display: inline-block;"><polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/></svg>''',
        "download": f'''<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; display: inline-block;"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>''',
        "dot_green": f'''<svg width="{size}" height="{size}" viewBox="0 0 24 24" style="vertical-align: middle; display: inline-block;"><circle cx="12" cy="12" r="8" fill="#10B981"/></svg>''',
        "dot_red": f'''<svg width="{size}" height="{size}" viewBox="0 0 24 24" style="vertical-align: middle; display: inline-block;"><circle cx="12" cy="12" r="8" fill="#EF4444"/></svg>'''
    }
    return icons.get(name, "")

# ---------------------------------------------------------
# Custom UI Styling (CSS)
# ---------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Enforce #FAF9F6 Background Across Entire App */
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"], [data-testid="stToolbar"] {
        background-color: #FAF9F6 !important;
        color: #0F172A !important;
    }

    /* Sidebar Warm Off-White Styling */
    [data-testid="stSidebar"], section[data-testid="stSidebar"], [data-testid="stSidebarContent"] {
        background-color: #F8F8DA !important;
        border-right: 1px solid #E5E5C0 !important;
    }

    /* Base Typography Colors */
    h1, h2, h3, h4, h5, h6, .stMarkdown p, .stMarkdown span {
        color: #0F172A;
    }

    .stCaption, [data-testid="stCaptionContainer"] {
        color: #475569 !important;
    }

    /* Tabs Styling */
    button[data-baseweb="tab"] {
        color: #475569 !important;
        font-weight: 600 !important;
        background-color: transparent !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #0284C7 !important;
        border-bottom-color: #0284C7 !important;
        font-weight: 700 !important;
    }

    /* File Uploader Container */
    [data-testid="stFileUploader"] {
        background-color: #FFFFFF !important;
        border: 1px dashed #CBD5E1 !important;
        border-radius: 10px !important;
        padding: 12px !important;
    }
    [data-testid="stFileUploaderDropzone"] {
        background-color: #FFFFF0 !important;
    }
    [data-testid="stFileUploaderDropzone"] span, [data-testid="stFileUploaderDropzone"] div {
        color: #334155 !important;
    }

    /* Hero Header */
    .hero-container {
        background: linear-gradient(135deg, #0A192F 0%, #172A45 50%, #1E3A8A 100%) !important;
        border-radius: 14px;
        padding: 32px 36px;
        color: #FFFFFF !important;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px rgba(10, 25, 47, 0.15);
        border: 1px solid rgba(255, 255, 255, 0.12);
    }
    .hero-container * {
        color: #FFFFFF !important;
    }
    .hero-title-row {
        display: flex;
        align-items: center;
        gap: 14px;
        margin-bottom: 8px;
    }
    .hero-title {
        font-size: 2.1rem !important;
        font-weight: 800 !important;
        margin: 0 !important;
        letter-spacing: -0.4px !important;
        color: #FFFFFF !important;
        line-height: 1.2 !important;
    }
    .hero-subtitle {
        font-size: 1.02rem !important;
        color: #E2E8F0 !important;
        font-weight: 400 !important;
        line-height: 1.6 !important;
        max-width: 920px !important;
        margin-top: 10px !important;
        margin-bottom: 18px !important;
    }
    .hero-badges {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
    }
    .hero-badge {
        background: rgba(255, 255, 255, 0.15) !important;
        border: 1px solid rgba(255, 255, 255, 0.25) !important;
        border-radius: 20px !important;
        padding: 6px 14px !important;
        font-size: 0.82rem !important;
        font-weight: 600 !important;
        color: #FFFFFF !important;
        display: inline-flex !important;
        align-items: center !important;
        gap: 7px !important;
    }
    .hero-badge * {
        color: #FFFFFF !important;
    }

    /* Diagnosis Cards */
    .scan-card {
        background: #FFFFFF;
        border-radius: 12px;
        border: 1px solid #E2E8F0;
        padding: 22px;
        margin-bottom: 22px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.03);
    }
    
    /* Status Badges */
    .badge-pneumonia {
        background: #FEE2E2;
        color: #991B1B;
        border: 1px solid #FCA5A5;
        padding: 6px 14px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.92rem;
        display: inline-flex;
        align-items: center;
        gap: 7px;
        letter-spacing: 0.2px;
    }
    .badge-normal {
        background: #DCFCE7;
        color: #166534;
        border: 1px solid #86EFAC;
        padding: 6px 14px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.92rem;
        display: inline-flex;
        align-items: center;
        gap: 7px;
        letter-spacing: 0.2px;
    }
    .badge-error {
        background: #FEF3C7;
        color: #92400E;
        border: 1px solid #FCD34D;
        padding: 6px 14px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.92rem;
    }

    /* Consensus Alert Banners */
    .consensus-alert-danger {
        background: #FEF2F2;
        border-left: 4px solid #DC2626;
        border-radius: 6px;
        padding: 12px 16px;
        margin: 12px 0 16px 0;
        color: #7F1D1D;
        font-size: 0.92rem;
        line-height: 1.5;
    }
    .consensus-alert-success {
        background: #F0FDF4;
        border-left: 4px solid #16A34A;
        border-radius: 6px;
        padding: 12px 16px;
        margin: 12px 0 16px 0;
        color: #14532D;
        font-size: 0.92rem;
        line-height: 1.5;
    }
    .consensus-alert-warning {
        background: #FFFBEB;
        border-left: 4px solid #D97706;
        border-radius: 6px;
        padding: 12px 16px;
        margin: 12px 0 16px 0;
        color: #78350F;
        font-size: 0.92rem;
        line-height: 1.5;
    }

    /* Metric Summary KPI Cards */
    .metric-box {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 18px 20px;
        text-align: center;
    }
    .metric-value {
        font-size: 2.1rem;
        font-weight: 800;
        color: #0F172A;
        line-height: 1.2;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        font-weight: 700;
        margin-top: 4px;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 6px;
    }

    /* Subcards for Individual Models */
    .model-subcard {
        background: #F8FAFC;
        border-radius: 8px;
        border: 1px solid #E2E8F0;
        padding: 14px 16px;
        margin-bottom: 12px;
    }
    .model-name {
        font-size: 0.8rem;
        font-weight: 700;
        color: #475569;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        gap: 6px;
    }

    /* Confidence Meters */
    .confidence-wrapper {
        margin-top: 10px;
    }
    .confidence-header {
        display: flex;
        justify-content: space-between;
        font-size: 0.78rem;
        color: #475569;
        margin-bottom: 5px;
        font-weight: 600;
    }
    .confidence-track {
        background: #E2E8F0;
        border-radius: 4px;
        height: 7px;
        overflow: hidden;
    }
    .confidence-fill-danger {
        background: #DC2626;
        height: 100%;
        border-radius: 4px;
    }
    .confidence-fill-success {
        background: #16A34A;
        height: 100%;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Path Handling & Model Caching
# ---------------------------------------------------------
def get_model_path(filename):
    """Locate model path relative to root or app folder."""
    p1 = os.path.join("models", filename)
    if os.path.exists(p1):
        return p1
    script_dir = os.path.dirname(os.path.abspath(__file__))
    p2 = os.path.join(script_dir, "..", "models", filename)
    if os.path.exists(p2):
        return p2
    p3 = os.path.join(script_dir, "models", filename)
    if os.path.exists(p3):
        return p3
    return p1

def get_samples_dir():
    """Locate images_test folder."""
    p1 = "images_test"
    if os.path.isdir(p1):
        return p1
    script_dir = os.path.dirname(os.path.abspath(__file__))
    p2 = os.path.join(script_dir, "..", "images_test")
    if os.path.isdir(p2):
        return p2
    return None

@st.cache_resource(show_spinner=False)
def load_all_models():
    """Cached loading of CNN and Logistic Regression models."""
    lr_file = get_model_path('logistic_regression_model.pkl')
    cnn_file = get_model_path('cnn_model.h5')
    
    with open(lr_file, 'rb') as f:
        lr = pickle.load(f)
    cnn = load_model(cnn_file)
    return lr, cnn

# Load the models
try:
    lr_model, cnn_model = load_all_models()
    models_ready = True
except Exception as e:
    st.error(f"Error loading models: {e}")
    models_ready = False

# ---------------------------------------------------------
# Core Image Preprocessing (EXACT SAME PREPROCESSING LOGIC)
# ---------------------------------------------------------
def preprocess_image(image):
    try:
        # Convert image to grayscale if it's RGB
        if image.mode == "RGB":
            image = image.convert("L")
        
        # Resize the image to 150x150 (input size for the models)
        image = image.resize((150, 150))
        
        # Convert image to a numpy array and normalize pixel values
        img_array = np.array(image) / 255.0

        # Reshape for CNN (add batch and channel dimensions)
        img_cnn = img_array.reshape(1, 150, 150, 1)

        # Flatten the image for Logistic Regression (22500 features)
        img_flat = img_array.flatten().reshape(1, -1)

        return img_cnn, img_flat
    except Exception as e:
        st.error(f"Error processing the image: {e}")
        return None, None

# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
with st.sidebar:
    st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 18px;">
            <div style="background: #0F172A; padding: 8px; border-radius: 8px; display: inline-flex;">
                {get_svg_icon("lungs", color="#38BDF8", size=24)}
            </div>
            <div>
                <h3 style="margin: 0; font-size: 1.18rem; font-weight: 800; color: #0F172A;">PneumoScan AI</h3>
                <span style="font-size: 0.75rem; background: #E0F2FE; color: #0369A1; padding: 2px 7px; border-radius: 4px; font-weight: 700;">Clinical Edition</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("### Sample Chest Radiographs")
    st.caption("Evaluate system performance immediately using verified test cases:")

    samples_dir = get_samples_dir()
    sample_files = []
    if samples_dir and os.path.exists(samples_dir):
        sample_files = [f for f in os.listdir(samples_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if "loaded_samples" not in st.session_state:
        st.session_state.loaded_samples = []

    if sample_files:
        sample_choices = {
            "NORMAL2-IM-0401-0001.jpeg": "Normal Radiograph (Subject A)",
            "NORMAL2-IM-0775-0001.jpeg": "Normal Radiograph (Subject B)",
            "person1712_bacteria_4529.jpeg": "Bacterial Pneumonia (Subject C)",
            "person896_virus_1548.jpeg": "Viral Pneumonia (Subject D)",
        }
        available_presets = {k: v for k, v in sample_choices.items() if k in sample_files}

        selected_preset = st.selectbox(
            "Select test case:",
            options=list(available_presets.keys()),
            format_func=lambda k: available_presets[k]
        )

        col_sb1, col_sb2 = st.columns(2)
        with col_sb1:
            if st.button("Add Selected", use_container_width=True):
                full_path = os.path.join(samples_dir, selected_preset)
                if full_path not in [s['path'] for s in st.session_state.loaded_samples]:
                    img = Image.open(full_path)
                    st.session_state.loaded_samples.append({
                        "name": selected_preset,
                        "image": img,
                        "path": full_path
                    })
                    st.rerun()

        with col_sb2:
            if st.button("Load All (4)", use_container_width=True):
                st.session_state.loaded_samples = []
                for fname in available_presets.keys():
                    full_path = os.path.join(samples_dir, fname)
                    img = Image.open(full_path)
                    st.session_state.loaded_samples.append({
                        "name": fname,
                        "image": img,
                        "path": full_path
                    })
                st.rerun()

        if st.session_state.loaded_samples:
            if st.button("Clear Test Queue", use_container_width=True):
                st.session_state.loaded_samples = []
                st.rerun()

    st.markdown("---")

    # System Status
    st.markdown("### System Architecture")
    if models_ready:
        st.markdown(f"""
            <div style="background: #F0FDF4; border: 1px solid #BBF7D0; border-radius: 6px; padding: 12px; margin-bottom: 14px;">
                <div style="display: flex; align-items: center; gap: 8px; font-weight: 700; color: #166534; font-size: 0.84rem;">
                    {get_svg_icon("check", color="#166534", size=16)} Active Diagnostic Pipeline
                </div>
                <div style="font-size: 0.78rem; color: #15803D; margin-top: 5px; line-height: 1.4;">
                    • CNN Model (150×150 Tensor)<br/>
                    • Logistic Regression (22,500 Vector)
                </div>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.error("Diagnostic models not found.")

    st.markdown("### Protocol Summary")
    st.markdown("""
    1. Upload digital chest radiographs.
    2. Execute dual-model classification.
    3. Review consensus verdict & confidence metrics.
    4. Export batch diagnostic logs.
    """)

    st.markdown("---")
    st.markdown(f"""
        <div style="background: #FEF3C7; border: 1px solid #FDE68A; border-radius: 6px; padding: 10px; font-size: 0.74rem; color: #92400E; line-height: 1.4;">
            <strong>{get_svg_icon("alert", color="#92400E", size=13)} Research Protocol Notice:</strong><br/>
            This software is provided for academic, machine learning benchmarking, and screening demonstration purposes only.
        </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------
# Hero Banner
# ---------------------------------------------------------
st.markdown(f"""
    <div class="hero-container">
        <div class="hero-title-row">
            <span style="background: rgba(255,255,255,0.12); padding: 8px 10px; border-radius: 8px; display: inline-flex;">
                {get_svg_icon("lungs", color="#38BDF8", size=30)}
            </span>
            <div class="hero-title" style="color: #FFFFFF !important; font-size: 2.1rem; font-weight: 800; margin: 0; letter-spacing: -0.4px; line-height: 1.2;">Chest Radiograph Pneumonia Detection</div>
        </div>
        <div class="hero-subtitle" style="color: #E2E8F0 !important; font-size: 1.02rem; font-weight: 400; line-height: 1.6; max-width: 920px; margin-top: 8px; margin-bottom: 18px;">
            Automated clinical screening of chest X-ray imagery powered by Convolutional Neural Networks (CNN) 
            cross-validated with baseline Logistic Regression for verified consensus diagnosis.
        </div>
        <div class="hero-badges">
            <span class="hero-badge" style="color: #FFFFFF !important;">{get_svg_icon("cpu", color="#93C5FD", size=15)} <span style="color: #FFFFFF !important;">Dual-Model Inference</span></span>
            <span class="hero-badge" style="color: #FFFFFF !important;">{get_svg_icon("activity", color="#93C5FD", size=15)} <span style="color: #FFFFFF !important;">150×150 Matrix Analysis</span></span>
            <span class="hero-badge" style="color: #FFFFFF !important;">{get_svg_icon("layers", color="#93C5FD", size=15)} <span style="color: #FFFFFF !important;">Multi-Scan Batch Support</span></span>
            <span class="hero-badge" style="color: #FFFFFF !important;">{get_svg_icon("download", color="#93C5FD", size=15)} <span style="color: #FFFFFF !important;">Exportable Clinical Summary</span></span>
        </div>
    </div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Tabs Structure (No Emojis)
# ---------------------------------------------------------
tab_scan, tab_batch, tab_models = st.tabs([
    "Diagnostic Screening", 
    "Batch Analytics & Export", 
    "Model Architecture & Pipeline"
])

# ---------------------------------------------------------
# Tab 1: Diagnostic Screening
# ---------------------------------------------------------
with tab_scan:
    st.markdown("#### Input Radiographic Scans")
    
    col_up1, col_up2 = st.columns([3, 1])
    with col_up1:
        uploaded_files = st.file_uploader(
            "Upload chest X-ray images (.jpg, .jpeg, .png)", 
            type=["jpg", "png", "jpeg"], 
            accept_multiple_files=True,
            help="Select one or multiple digital radiographs to classify."
        )
    
    with col_up2:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        active_sample_count = len(st.session_state.loaded_samples)
        if active_sample_count > 0:
            st.info(f"{active_sample_count} sample radiograph(s) loaded from sidebar queue.")
        else:
            st.caption("Tip: Select pre-loaded sample scans from the sidebar.")

    # Combine active images
    all_images = []
    
    if uploaded_files:
        for file in uploaded_files:
            try:
                img = Image.open(file)
                all_images.append({
                    "name": file.name,
                    "image": img,
                    "source": "Upload"
                })
            except Exception as e:
                st.error(f"Error loading file {file.name}: {e}")

    for s in st.session_state.loaded_samples:
        all_images.append({
            "name": s["name"],
            "image": s["image"],
            "source": "Sample"
        })

    # Display preview gallery
    if all_images:
        st.markdown("---")
        st.markdown(f"#### Active Radiograph Queue ({len(all_images)} scans ready)")
        
        cols_per_row = 4
        cols = st.columns(min(len(all_images), cols_per_row))
        for i, item in enumerate(all_images):
            with cols[i % len(cols)]:
                source_label = "Sample" if item.get("source") == "Sample" else "Upload"
                st.image(
                    item["image"], 
                    caption=f"Scan #{i + 1}: {item['name'][:18]}... [{source_label}]", 
                    use_container_width=True
                )

        st.markdown("---")
        
        # Action Buttons
        col_act1, col_act2, col_act3 = st.columns([2, 1, 1])
        with col_act1:
            run_button = st.button("Run Diagnostic Classification", type="primary", use_container_width=True)
        with col_act2:
            if st.button("Reset Analysis", use_container_width=True):
                st.session_state.loaded_samples = []
                if "results_data" in st.session_state:
                    del st.session_state["results_data"]
                st.rerun()

        # Classification Execution
        if run_button or ("results_data" in st.session_state and st.session_state.get("active_img_count") == len(all_images)):
            if run_button:
                progress_bar = st.progress(0)
                status_text = st.empty()
                results = []

                for i, item in enumerate(all_images):
                    img = item["image"]
                    status_text.text(f"Processing scan {i + 1} of {len(all_images)}: {item['name']}...")
                    progress_bar.progress((i + 1) / len(all_images))
                    
                    # Exact Preprocessing Logic
                    img_cnn, img_flat = preprocess_image(img)

                    if img_cnn is not None and img_flat is not None:
                        # CNN Prediction (EXACT SAME LOGIC)
                        cnn_raw = cnn_model.predict(img_cnn, verbose=0)
                        cnn_preds = (cnn_raw > 0.5).astype(int)

                        # Logistic Regression Prediction (EXACT SAME LOGIC)
                        lr_preds = lr_model.predict(img_flat)
                        
                        # Labels
                        cnn_label = "Pneumonia" if cnn_preds == 0 else "Normal"
                        lr_label = "Pneumonia" if lr_preds == 0 else "Normal"

                        # Confidence Calculation
                        cnn_score_val = float(cnn_raw[0][0])
                        if cnn_preds == 0:
                            cnn_confidence = (1.0 - cnn_score_val) * 100.0
                        else:
                            cnn_confidence = cnn_score_val * 100.0

                        lr_confidence = 100.0
                        if hasattr(lr_model, "predict_proba"):
                            try:
                                lr_prob = lr_model.predict_proba(img_flat)[0]
                                lr_confidence = float(lr_prob[0] if lr_preds == 0 else lr_prob[1]) * 100.0
                            except Exception:
                                lr_confidence = 100.0

                        results.append({
                            "Image": f"Scan #{i + 1}",
                            "File Name": item["name"],
                            "Source": item.get("source", "Upload"),
                            "CNN": cnn_label,
                            "CNN Confidence": cnn_confidence,
                            "Logistic Regression": lr_label,
                            "LR Confidence": lr_confidence,
                            "Agreement": "Yes" if cnn_label == lr_label else "No",
                            "image_obj": img,
                            "dimensions": f"{img.width} × {img.height}"
                        })
                    else:
                        results.append({
                            "Image": f"Scan #{i + 1}",
                            "File Name": item["name"],
                            "Source": item.get("source", "Upload"),
                            "CNN": "Error in processing",
                            "CNN Confidence": 0.0,
                            "Logistic Regression": "Error in processing",
                            "LR Confidence": 0.0,
                            "Agreement": "N/A",
                            "image_obj": img,
                            "dimensions": "N/A"
                        })

                status_text.empty()
                progress_bar.empty()
                
                st.session_state["results_data"] = results
                st.session_state["active_img_count"] = len(all_images)

            results = st.session_state.get("results_data", [])

            st.markdown("---")
            st.markdown("#### Diagnostic Findings & Consensus Review")
            
            for idx, res in enumerate(results):
                st.markdown('<div class="scan-card">', unsafe_allow_html=True)
                
                card_col1, card_col2 = st.columns([1, 2])
                
                with card_col1:
                    st.image(res["image_obj"], caption=f"{res['File Name']} ({res['dimensions']})", use_container_width=True)
                
                with card_col2:
                    st.markdown(f"### {res['Image']}: {res['File Name']}")
                    
                    # Consensus Banner
                    if res["CNN"] == "Pneumonia" and res["Logistic Regression"] == "Pneumonia":
                        st.markdown(f"""
                            <div class="consensus-alert-danger">
                                <strong>{get_svg_icon("shield_alert", color="#DC2626", size=18)} HIGH RISK CONSENSUS (PNEUMONIA DETECTED):</strong><br/>
                                Both the Deep CNN and Logistic Regression models independently diagnosed this scan as <strong>Pneumonia</strong>.
                            </div>
                        """, unsafe_allow_html=True)
                    elif res["CNN"] == "Normal" and res["Logistic Regression"] == "Normal":
                        st.markdown(f"""
                            <div class="consensus-alert-success">
                                <strong>{get_svg_icon("shield_check", color="#16A34A", size=18)} LOW RISK CONSENSUS (NORMAL):</strong><br/>
                                Both models concordantly classified this radiograph as <strong>Normal</strong> lung parenchymal tissue.
                            </div>
                        """, unsafe_allow_html=True)
                    elif "Error" in res["CNN"]:
                        st.markdown(f"""
                            <div class="consensus-alert-warning">
                                <strong>{get_svg_icon("alert", color="#D97706", size=18)} Processing Error:</strong> Radiograph could not be processed.
                            </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                            <div class="consensus-alert-warning">
                                <strong>{get_svg_icon("alert", color="#D97706", size=18)} DIVERGENT MODEL OUTCOMES:</strong><br/>
                                The CNN and Logistic Regression models provided conflicting classifications. Secondary radiological inspection is recommended.
                            </div>
                        """, unsafe_allow_html=True)

                    # Subcards for each model
                    subcol1, subcol2 = st.columns(2)
                    
                    with subcol1:
                        cnn_badge_class = "badge-pneumonia" if res["CNN"] == "Pneumonia" else ("badge-normal" if res["CNN"] == "Normal" else "badge-error")
                        cnn_icon = get_svg_icon("alert", color="#991B1B", size=15) if res["CNN"] == "Pneumonia" else get_svg_icon("check", color="#166534", size=15)
                        cnn_fill_class = "confidence-fill-danger" if res["CNN"] == "Pneumonia" else "confidence-fill-success"
                        
                        st.markdown(f"""
                            <div class="model-subcard">
                                <div class="model-name">
                                    {get_svg_icon("cpu", color="#475569", size=14)} Convolutional Neural Network
                                </div>
                                <div class="{cnn_badge_class}">
                                    {cnn_icon} {res['CNN']}
                                </div>
                                <div class="confidence-wrapper">
                                    <div class="confidence-header">
                                        <span>Model Confidence</span>
                                        <span>{res['CNN Confidence']:.1f}%</span>
                                    </div>
                                    <div class="confidence-track">
                                        <div class="{cnn_fill_class}" style="width: {res['CNN Confidence']:.1f}%;"></div>
                                    </div>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)

                    with subcol2:
                        lr_badge_class = "badge-pneumonia" if res["Logistic Regression"] == "Pneumonia" else ("badge-normal" if res["Logistic Regression"] == "Normal" else "badge-error")
                        lr_icon = get_svg_icon("alert", color="#991B1B", size=15) if res["Logistic Regression"] == "Pneumonia" else get_svg_icon("check", color="#166534", size=15)
                        lr_fill_class = "confidence-fill-danger" if res["Logistic Regression"] == "Pneumonia" else "confidence-fill-success"
                        
                        st.markdown(f"""
                            <div class="model-subcard">
                                <div class="model-name">
                                    {get_svg_icon("activity", color="#475569", size=14)} Logistic Regression
                                </div>
                                <div class="{lr_badge_class}">
                                    {lr_icon} {res['Logistic Regression']}
                                </div>
                                <div class="confidence-wrapper">
                                    <div class="confidence-header">
                                        <span>Model Confidence</span>
                                        <span>{res['LR Confidence']:.1f}%</span>
                                    </div>
                                    <div class="confidence-track">
                                        <div class="{lr_fill_class}" style="width: {res['LR Confidence']:.1f}%;"></div>
                                    </div>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)

                    with st.expander("Technical Findings & Dimensional Metrics"):
                        if res["CNN"] == "Pneumonia":
                            st.write("• **Radiological Pattern**: Bilateral or focal infiltrates, opacity, or consolidation detected.")
                        else:
                            st.write("• **Radiological Pattern**: Clear pulmonary parenchyma without focal consolidation.")
                        st.caption(f"Input tensor: 150 × 150 grayscale normalized [0.0, 1.0]. Feature vector: 22,500 dimensions.")

                st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("Upload one or more chest X-ray images above, or add sample scans from the sidebar to begin.")

# ---------------------------------------------------------
# Tab 2: Batch Analytics & Export
# ---------------------------------------------------------
with tab_batch:
    st.markdown("### Batch Classification Metrics")
    
    if "results_data" in st.session_state and st.session_state["results_data"]:
        results = st.session_state["results_data"]
        
        total_scans = len(results)
        cnn_pneumonia_count = sum(1 for r in results if r["CNN"] == "Pneumonia")
        cnn_normal_count = sum(1 for r in results if r["CNN"] == "Normal")
        agreed_count = sum(1 for r in results if r["Agreement"] == "Yes")
        agreement_rate = (agreed_count / total_scans * 100.0) if total_scans > 0 else 0.0

        # KPI Metrics Row
        mcol1, mcol2, mcol3, mcol4 = st.columns(4)
        with mcol1:
            st.markdown(f'<div class="metric-box"><div class="metric-value">{total_scans}</div><div class="metric-label">{get_svg_icon("file", color="#64748B", size=14)} Total Scans</div></div>', unsafe_allow_html=True)
        with mcol2:
            st.markdown(f'<div class="metric-box"><div class="metric-value" style="color: #DC2626;">{cnn_pneumonia_count}</div><div class="metric-label">{get_svg_icon("alert", color="#DC2626", size=14)} Pneumonia Cases</div></div>', unsafe_allow_html=True)
        with mcol3:
            st.markdown(f'<div class="metric-box"><div class="metric-value" style="color: #16A34A;">{cnn_normal_count}</div><div class="metric-label">{get_svg_icon("check", color="#16A34A", size=14)} Normal Scans</div></div>', unsafe_allow_html=True)
        with mcol4:
            st.markdown(f'<div class="metric-box"><div class="metric-value" style="color: #2563EB;">{agreement_rate:.0f}%</div><div class="metric-label">{get_svg_icon("shield_check", color="#2563EB", size=14)} Model Consensus</div></div>', unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("#### Diagnostic Summary Table")

        # Build custom styled table (No pyarrow dependency, zero indent to prevent code block)
        table_rows = []
        html_rows = []
        for r in results:
            cnn_badge = f'<span class="badge-pneumonia" style="padding: 3px 8px; font-size: 0.78rem;">{r["CNN"]}</span>' if r["CNN"] == "Pneumonia" else f'<span class="badge-normal" style="padding: 3px 8px; font-size: 0.78rem;">{r["CNN"]}</span>'
            lr_badge = f'<span class="badge-pneumonia" style="padding: 3px 8px; font-size: 0.78rem;">{r["Logistic Regression"]}</span>' if r["Logistic Regression"] == "Pneumonia" else f'<span class="badge-normal" style="padding: 3px 8px; font-size: 0.78rem;">{r["Logistic Regression"]}</span>'
            consensus_badge = '<span style="background: #DCFCE7; color: #166534; border: 1px solid #86EFAC; padding: 3px 8px; border-radius: 4px; font-weight: 700; font-size: 0.78rem;">Concordant</span>' if r["Agreement"] == "Yes" else '<span style="background: #FEF3C7; color: #92400E; border: 1px solid #FCD34D; padding: 3px 8px; border-radius: 4px; font-weight: 700; font-size: 0.78rem;">Discordant</span>'
            
            html_rows.append(
                f'<tr style="border-bottom: 1px solid #E2E8F0;">'
                f'<td style="padding: 10px 14px; font-weight: 700; color: #0F172A;">{r["Image"]}</td>'
                f'<td style="padding: 10px 14px; color: #334155;">{r["File Name"]}</td>'
                f'<td style="padding: 10px 14px;">{cnn_badge}</td>'
                f'<td style="padding: 10px 14px; font-weight: 600; color: #0F172A;">{r["CNN Confidence"]:.2f}%</td>'
                f'<td style="padding: 10px 14px;">{lr_badge}</td>'
                f'<td style="padding: 10px 14px; font-weight: 600; color: #0F172A;">{r["LR Confidence"]:.2f}%</td>'
                f'<td style="padding: 10px 14px;">{consensus_badge}</td>'
                f'</tr>'
            )
            table_rows.append({
                "Scan Identifier": r["Image"],
                "File Name": r["File Name"],
                "CNN Diagnosis": r["CNN"],
                "CNN Confidence": f"{r['CNN Confidence']:.2f}%",
                "LR Diagnosis": r["Logistic Regression"],
                "LR Confidence": f"{r['LR Confidence']:.2f}%",
                "Model Consensus": "Concordant" if r["Agreement"] == "Yes" else "Discordant"
            })
        
        body_rows_markup = "".join(html_rows)
        custom_table_html = (
            '<div style="overflow-x: auto; border: 1px solid #E2E8F0; border-radius: 8px; margin-bottom: 20px; background: #FFFFFF;">'
            '<table style="width: 100%; border-collapse: collapse; font-size: 0.85rem; text-align: left;">'
            '<thead>'
            '<tr style="background-color: #F8FAFC; border-bottom: 2px solid #E2E8F0; color: #475569;">'
            '<th style="padding: 12px 14px; font-weight: 700;">Scan</th>'
            '<th style="padding: 12px 14px; font-weight: 700;">File Name</th>'
            '<th style="padding: 12px 14px; font-weight: 700;">CNN Verdict</th>'
            '<th style="padding: 12px 14px; font-weight: 700;">CNN Confidence</th>'
            '<th style="padding: 12px 14px; font-weight: 700;">LR Verdict</th>'
            '<th style="padding: 12px 14px; font-weight: 700;">LR Confidence</th>'
            '<th style="padding: 12px 14px; font-weight: 700;">Consensus</th>'
            '</tr>'
            '</thead>'
            f'<tbody>{body_rows_markup}</tbody>'
            '</table>'
            '</div>'
        )
        st.markdown(custom_table_html, unsafe_allow_html=True)

        df_results = pd.DataFrame(table_rows)

        # Download Report CSV
        csv_buffer = io.StringIO()
        df_results.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue()

        st.download_button(
            label="Download Diagnostic Report (.csv)",
            data=csv_data,
            file_name="pneumonia_diagnostic_report.csv",
            mime="text/csv",
            type="primary"
        )
    else:
        st.info("No classification data recorded yet. Upload radiographs or load sample scans in the first tab and click 'Run Diagnostic Classification'.")

# ---------------------------------------------------------
# Tab 3: Model Architecture & Benchmarks
# ---------------------------------------------------------
with tab_models:
    st.markdown("### Architecture & Preprocessing Pipeline")
    
    st.markdown("""
    This system combines deep convolutional spatial representations with regularized linear boundaries 
    for dual-perspective pneumonia detection.
    """)

    col_arch1, col_arch2 = st.columns(2)
    
    with col_arch1:
        st.markdown(f"""
        <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 10px; padding: 20px;">
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
                {get_svg_icon("cpu", color="#0F172A", size=20)}
                <h4 style="color: #0F172A; margin: 0;">1. Convolutional Neural Network (CNN)</h4>
            </div>
            <p style="font-size: 0.88rem; color: #475569;">
                Deep hierarchical feature extraction tailored for medical image pattern recognition.
            </p>
            <ul style="font-size: 0.85rem; color: #334155; line-height: 1.6;">
                <li><strong>Input Layer:</strong> 150 × 150 × 1 (Single-channel Grayscale)</li>
                <li><strong>Architecture:</strong> Convolutional filters, ReLU activations, Max-Pooling</li>
                <li><strong>Output:</strong> Sigmoid activation unit</li>
                <li><strong>Decision Criterion:</strong> Score &gt; 0.5 &rarr; Normal (1), else Pneumonia (0)</li>
                <li><strong>Test Accuracy:</strong> ~94.8%</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col_arch2:
        st.markdown(f"""
        <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 10px; padding: 20px;">
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
                {get_svg_icon("activity", color="#0F172A", size=20)}
                <h4 style="color: #0F172A; margin: 0;">2. Logistic Regression Classifier</h4>
            </div>
            <p style="font-size: 0.88rem; color: #475569;">
                Linear decision boundary providing a baseline statistical benchmark.
            </p>
            <ul style="font-size: 0.85rem; color: #334155; line-height: 1.6;">
                <li><strong>Input Layer:</strong> 22,500 Flattened pixel intensities normalized to [0, 1]</li>
                <li><strong>Regularization:</strong> L2 Regularized maximum likelihood optimization</li>
                <li><strong>Output:</strong> Binary class label (0: Pneumonia, 1: Normal)</li>
                <li><strong>Role:</strong> Concordance benchmark against deep learning model</li>
                <li><strong>Test Accuracy:</strong> ~86.2%</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### Standardization & Preprocessing Pipeline")
    
    step_col1, step_col2, step_col3, step_col4 = st.columns(4)
    with step_col1:
        st.markdown("""
            <div style="background: #F1F5F9; border-radius: 8px; padding: 14px; text-align: center;">
                <div style="font-weight: 700; color: #1E293B; font-size: 0.88rem;">Step 1: Color Space</div>
                <div style="font-size: 0.78rem; color: #64748B; margin-top: 4px;">Convert RGB to Grayscale (L)</div>
            </div>
        """, unsafe_allow_html=True)
    with step_col2:
        st.markdown("""
            <div style="background: #F1F5F9; border-radius: 8px; padding: 14px; text-align: center;">
                <div style="font-weight: 700; color: #1E293B; font-size: 0.88rem;">Step 2: Resampling</div>
                <div style="font-size: 0.78rem; color: #64748B; margin-top: 4px;">Standardize to 150 × 150 pixels</div>
            </div>
        """, unsafe_allow_html=True)
    with step_col3:
        st.markdown("""
            <div style="background: #F1F5F9; border-radius: 8px; padding: 14px; text-align: center;">
                <div style="font-weight: 700; color: #1E293B; font-size: 0.88rem;">Step 3: Normalization</div>
                <div style="font-size: 0.78rem; color: #64748B; margin-top: 4px;">Normalize pixel intensities [0.0, 1.0]</div>
            </div>
        """, unsafe_allow_html=True)
    with step_col4:
        st.markdown("""
            <div style="background: #F1F5F9; border-radius: 8px; padding: 14px; text-align: center;">
                <div style="font-weight: 700; color: #1E293B; font-size: 0.88rem;">Step 4: Tensor Reshape</div>
                <div style="font-size: 0.78rem; color: #64748B; margin-top: 4px;">CNN: (1, 150, 150, 1) | LR: (1, 22500)</div>
            </div>
        """, unsafe_allow_html=True)
