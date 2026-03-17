import streamlit as st
import cv2
import numpy as np
import pandas as pd
from analyzer import ParticleAnalyzer
from PIL import Image
import io

# Page Configuration
st.set_page_config(page_title="Agnostic Particle Analyzer", layout="wide")

# Custom CSS for High-Contrast Premium Look
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Outfit:wght@400;700&display=swap');

    /* 1. ELIMINATE TOP SPACE & STREAMLIT UI */
    [data-testid="stHeader"] {
        display: none !important;
    }
    
    #root > div:nth-child(1) > div > div > div > div > section > div {
        padding-top: 1rem !important;
    }

    /* 2. ABSOLUTE VISIBILITY OVERRIDES (FORCED WHITE) */
    /* Force main dashboard and sidebar content to pure white */
    .main .stMarkdown p, .main .stMarkdown li, .main .stMarkdown span,
    .main label, .main h1, .main h2, .main h3, .main h4,
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] li, 
    [data-testid="stSidebar"] label, [data-testid="stSidebar"] h3 {
        color: #FFFFFF !important;
        font-weight: 500;
    }

    /* Force specific hero-box and subtitle visibility */
    .hero-box * {
        color: #FFFFFF !important;
    }

    /* Fix status messages (Analysis Successful) visibility */
    [data-testid="stStatus"] * {
        color: #FFFFFF !important;
    }

    /* 3. INTERACTIVE INPUT CONTRAST (BLACK ON WHITE) */
    /* These elements have light backgrounds, so text must be black */
    input, select, textarea, [data-baseweb="select"] * {
        color: #000000 !important;
    }

    [data-testid="stAppViewContainer"] {
        background-color: #0f172a !important; 
    }
    
    .main {
        background-color: #0f172a !important;
    }

    /* 4. TYPOGRAPHY & HEADINGS */
    h1, h2, h3, h4 {
        font-family: 'Outfit', sans-serif;
    }

    .project-title {
        font-size: 3.5rem !important;
        font-weight: 700 !important;
        margin-top: 20px;
        background: linear-gradient(to right, #ffffff, #34d399); 
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent !important;
    }

    h2, h3 {
        color: #34d399 !important; /* Brighter emerald for labels */
        margin-top: 2rem !important;
    }

    /* 5. SIDEBAR - PURE DARK NAVY */
    [data-testid="stSidebar"] {
        background-color: #020617 !important;
        border-right: 2px solid #334155;
    }
    
    [data-testid="stSidebar"] section {
        padding-top: 2rem;
    }

    /* Sidebar selectbox high-contrast area */
    [data-testid="stSidebar"] div[data-baseweb="select"] {
        background-color: #ffffff !important;
        border: 2px solid #34d399 !important;
        border-radius: 8px !important;
    }
    
    /* File uploader high-contrast area */
    [data-testid="stFileUploader"] section {
        background-color: #ffffff !important;
        border: 2px dashed #34d399 !important;
        border-radius: 12px !important;
    }
    
    [data-testid="stFileUploader"] section * {
        color: #000000 !important;
    }

    /* 6. METRICS & CARDS */
    .stMetric {
        background: #1e293b !important; /* High contrast card */
        border: 2px solid #334155 !important;
        border-radius: 12px !important;
        padding: 20px !important;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.5) !important;
    }
    
    .stMetric:hover {
        border: 2px solid #10b981 !important;
    }

    [data-testid="stMetricValue"] {
        color: #10b981 !important;
        font-size: 2.8rem !important;
        font-weight: 700 !important;
    }

    [data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
        font-weight: 500 !important;
        font-size: 0.9rem !important;
    }

    /* 6. BUTTONS & ACTIONS */
    .stButton>button {
        background: #10b981 !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.75rem 2rem !important;
        font-weight: 700 !important;
        width: 100%;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .stButton>button:hover {
        background: #34d399 !important;
        box-shadow: 0 0 20px rgba(16, 185, 129, 0.5) !important;
    }

    /* 7. TABLES & DATAFRAMES (FORCE DARK MODE) */
    [data-testid="stDataFrame"] {
        background-color: #0f172a !important;
        border: 1px solid #334155 !important;
        border-radius: 12px !important;
    }
    
    /* EXTREME FIX: Dataframe Toolbar (Download/Zoom) Visibility */
    [data-testid="stElementToolbar"], 
    [data-testid="stElementToolbar"] div,
    [data-testid="stElementToolbar"] p {
        background-color: #020617 !important; /* Dark Navy */
        border-radius: 8px !important;
    }

    [data-testid="stElementToolbar"] button, 
    [data-testid="stElementToolbar"] svg,
    [data-testid="stElementToolbar"] span {
        color: #34d399 !important; /* Vibrant Emerald */
        fill: #34d399 !important;
        stroke: #34d399 !important;
    }

    /* High-Contrast Hover State */
    [data-testid="stElementToolbar"] button:hover {
        background-color: #1e293b !important;
        box-shadow: 0 0 10px rgba(52, 211, 153, 0.6) !important;
    }

    [data-testid="stElementToolbar"] button:hover svg {
        color: #ffffff !important;
        fill: #ffffff !important;
    }

    /* Center the dataframe container */
    [data-testid="stColumn"] > div > div > div > [data-testid="stDataFrame"] {
        margin-left: auto !important;
        margin-right: auto !important;
    }

    /* Ensure table content is bright */
    [data-testid="stTable"], [data-testid="stDataFrame"] * {
        color: #ffffff !important;
    }

    /* Hero section */
    .hero-box {
        background: #1e293b;
        border-left: 8px solid #10b981;
        padding: 2rem;
        border-radius: 0 16px 16px 0;
        margin-bottom: 3rem;
        box-shadow: 4px 4px 20px rgba(0,0,0,0.3);
    }
    #dashboard-subtitle {
        color: #FFFFFF !important;
        font-size: 1.4rem !important;
        margin-top: -15px !important;
        opacity: 1 !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Main Navigation Sidebar
with st.sidebar:
    st.markdown("### SYSTEM CONTROLS")
    app_mode = st.selectbox("Switch Dashboard", ["Vision Analyzer", "System Methodology", "Data Testbench"], label_visibility="collapsed")
    st.divider()
    
    if app_mode == "Vision Analyzer":
        st.markdown("#### Feature Input")
        uploaded_file = st.file_uploader("Upload Micrograph", type=['jpg', 'jpeg', 'png', 'tiff'])

# Main Dashboard Layout
st.markdown('<h1 class="project-title">MorphoVision</h1>', unsafe_allow_html=True)
st.markdown('<p id="dashboard-subtitle">Automated Particle Morphology & Metrics Dashboard</p>', unsafe_allow_html=True)

st.markdown("""
<div class="hero-box">
    <h3 style="margin-top:0 !important; color:#ffffff !important;">High-Contrast Adaptive Intelligence</h3>
    <p style="margin-bottom:0 !important;">
        Integrating <b>"Zero-Parameter"</b> neural logic with material-agnostic segmentation. 
        Designed for extreme precision in research and development morphology datasets.
    </p>
</div>
""", unsafe_allow_html=True)

if app_mode == "Vision Analyzer":
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        analyzer = ParticleAnalyzer()
        
        with st.status("🚀 Processing Data Streams...", expanded=True) as status:
            st.write("Initializing Engine...")
            binary = analyzer.preprocess(image)
            st.write("Applying Neural Watershed...")
            labels = analyzer.segment(binary)
            st.write("Extracting Morphological Tensors...")
            df, pai = analyzer.calculate_metrics(labels, binary)
            output_img = analyzer.get_colored_output(image, labels, df)
            status.update(label="Analysis Successfully Sealed", state="complete", expanded=False)
            
        st.markdown("## Feature Visualization")
        
        # Row 1: Image and Key Metrics
        col_img, col_metrics = st.columns([1.2, 0.8])
        
        with col_img:
            st.markdown('<div style="background:#020617; border:1px solid #334155; border-radius:12px; padding:10px;">', unsafe_allow_html=True)
            st.image(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB), use_container_width=True, 
                     caption="Output Mapping: Green (Isolated) | Red (Aggregated/Clusters) | Yellow (Edge Case)")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col_metrics:
            st.metric("Total Count", len(df))
            st.metric("PAI Ratio", f"{pai:.2f}")
            st.metric("Avg Complexity", f"{df['Complexity'].mean():.2f}")
            
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("EXPORT CSV PACKET", data=csv, file_name='morphovision_profile.csv', mime='text/csv')

        # Row 2: Centered Results Table
        st.divider()
        st.markdown("<h2 style='text-align: center;'>Primary Data Packet</h2>", unsafe_allow_html=True)
        _, table_col, _ = st.columns([0.1, 0.8, 0.1])
        with table_col:
            st.dataframe(df[['Area', 'Complexity', 'State']], use_container_width=True, height=400)

        # Row 3: Statistical Analytics
        st.divider()
        st.markdown("## Statistical Engine")
        v_col1, v_col2 = st.columns(2)
        
        with v_col1:
            st.markdown("#### Particle Area Spread")
            st.bar_chart(df['Area'], height=250)
            
        with v_col2:
            st.markdown("#### Complexity Histogram")
            st.area_chart(df['Complexity'], height=250)

    else:
        st.info("Input a particle micrograph in the system controls (sidebar) to begin the analytical sequence.")
        st.image("https://images.unsplash.com/photo-1542382257-80dedb725088?auto=format&fit=crop&q=80&w=1200", caption="MorphoVision: High-Contrast Scientific Terminal", use_container_width=True)
        
elif app_mode == "System Methodology":
    st.markdown("## System Architecture")
    m_col1, m_col2 = st.columns(2)
    with m_col1:
        st.markdown("""
            ### 1. Zero-Parameter Core
            Adaptive **Otsu Binarization** combined with dynamic thresholding logic for multi-material micrographs.
            
            ### 2. Neural Watershed
            Solving the **Heavy Agglomeration Problem** via Euclidean transform markers and watershed-based delumping.
        """)
    with m_col2:
        st.markdown("""
            ### 3. Quantitative Indices
            - **PAI Index**: Cluster density quantification.
            - **Morpho-Complexity**: Shape deviation analysis.
        """)

elif app_mode == "Data Testbench":
    st.markdown("## Terminal Bench")
    if st.button("SYNTHESIZE TEST PACKET"):
        demo_img = np.zeros((400, 600, 3), dtype=np.uint8)
        noise = np.random.randint(0, 30, (400, 600, 3), dtype=np.uint8)
        demo_img = cv2.add(demo_img, noise)
        cv2.circle(demo_img, (150, 150), 30, (200, 200, 200), -1)
        cv2.circle(demo_img, (400, 250), 35, (200, 200, 200), -1)
        st.image(demo_img, caption="Synthetic Micrograph Stream", use_container_width=True)
        st.download_button("DOWNLOAD PACKET", data=io.BytesIO(cv2.imencode(".png", demo_img)[1]), file_name="sample.png", mime="image/png")
