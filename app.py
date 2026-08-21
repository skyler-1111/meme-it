import streamlit as st
import os
from PIL import Image
import time
import io

# Import our custom image processing algorithms
from src.color_transfer import transfer_color
from src.neural_morph import run_neural_morph
from src.style_transfer import run_style_transfer

# Set page configuration
st.set_page_config(
    page_title="img morpher",
    page_icon="🗿",
    layout="centered", # Centered layout is cleaner for a single column
    initial_sidebar_state="expanded"
)

# Custom Neo-Brutalist CSS styling (bright colors, thick borders, sharp shadows, bold styling)
st.markdown("""
<style>
/* Base App Container */
.stApp {
    background-color: #0d0d0d !important; /* Premium Ink Black */
    color: #f5f5f7 !important; /* Soft, elegant off-white */
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
    letter-spacing: -0.02em;
}

/* Elegant, Clean Header Card */
.app-title {
    text-align: center;
    background-color: #1a1a1a !important; /* Rich Dark Charcoal */
    color: #ffffff !important;
    border: 1px solid #2d2d2d !important; /* Thin, premium border */
    box-shadow: 0px 8px 30px rgba(0, 0, 0, 0.5) !important; /* Soft depth shadow */
    padding: 2rem !important;
    font-size: 2.5rem !important;
    font-weight: 300 !important; /* Light, elegant weight */
    letter-spacing: 0.05em !important; /* Sophisticated spacing */
    text-transform: uppercase;
    margin-top: 1.5rem !important;
    margin-bottom: 0.5rem !important;
}

.app-subtitle {
    text-align: center;
    color: #c5a880 !important; /* Elegant Muted Gold/Champagne accent */
    font-size: 0.9rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase;
    margin-bottom: 2.5rem !important;
    background-color: transparent !important; 
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
}

/* Sidebar Styling */
section[data-testid="stSidebar"] {
    background-color: #050505 !important; /* Ultra deep black for depth contrast */
    border-right: 1px solid #1c1c1e !important;
}

section[data-testid="stSidebar"] * {
    color: #e5e5ea !important;
}

/* Widget Labels styling */
div[data-testid="stWidgetLabel"] p {
    font-weight: 500 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    color: #a1a1aa !important; /* Muted gray for secondary importance */
}

/* Sidebar headers */
.sidebar-header {
    font-size: 1.1rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.1em;
    text-transform: uppercase !important;
    background-color: transparent !important; 
    color: #ffffff !important;
    padding: 0.5rem 0 !important;
    border-bottom: 1px solid #2d2d2d !important;
    margin-bottom: 1.5rem !important;
}

.sidebar-section-title {
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em;
    border-bottom: 1px solid #2d2d2d !important;
    margin-top: 2rem !important;
    margin-bottom: 1rem !important;
    padding-bottom: 0.4rem !important;
    color: #c5a880 !important; /* Subtle accent tie-in */
}

/* Clean, Sophisticated Upload Box Container */
.upload-card {
    background-color: #161617 !important;
    border: 1px solid #2c2c2e !important;
    padding: 2rem !important;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3) !important;
    margin-bottom: 2rem !important;
    border-radius: 4px;
}

.upload-card-title {
    font-size: 1rem !important;
    font-weight: 600 !important;
    color: #ffffff !important;
    background-color: transparent !important; 
    border: none !important;
    padding: 0 !important;
    margin-bottom: 1.2rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em;
    display: block;
    box-shadow: none !important;
}

/* Premium Action Trigger Button */
.stButton>button {
    background-color: #ffffff !important; 
    color: #000000 !important;
    border: 1px solid #ffffff !important;
    border-radius: 2px !important; /* Extremely slight roundness for high-end feel */
    padding: 0.8rem 2rem !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em;
    box-shadow: none !important;
    transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1) !important;
    width: 100% !important;
    margin-bottom: 2rem !important;
}

.stButton>button:hover {
    background-color: transparent !important;
    color: #ffffff !important;
    border-color: #ffffff !important;
    cursor: pointer;
}

.stButton>button:active {
    transform: scale(0.98) !important;
    background-color: rgba(255, 255, 255, 0.1) !important;
}

/* Morphed Display Container */
.morphed-card {
    background-color: #161617 !important;
    border: 1px solid #2c2c2e !important;
    padding: 2rem !important;
    box-shadow: 0 10px 30px rgba(0,0,0,0.4) !important;
    margin-top: 1.5rem !important;
    margin-bottom: 2rem !important;
    border-radius: 4px;
}

.morphed-card-title {
    font-size: 1rem !important;
    font-weight: 600 !important;
    color: #c5a880 !important;
    background-color: transparent !important; 
    border: none !important;
    padding: 0 !important;
    margin-bottom: 1.2rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em;
    display: block;
    box-shadow: none !important;
}

/* Stats & Progress Boxes */
.metrics-container {
    background-color: #1c1c1e !important;
    border: 1px solid #2c2c2e !important;
    padding: 1.2rem !important;
    box-shadow: none !important;
    color: #ffffff !important;
    margin-bottom: 1rem !important;
    border-radius: 4px;
}

.metrics-container h4 {
    color: #ffffff !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em;
    margin-top: 0 !important;
    border-bottom: 1px solid #2c2c2e !important;
    padding-bottom: 0.5rem !important;
}

.metrics-container p {
    color: #c5a880 !important; /* Gold text highlight for data metrics */
    font-weight: 300 !important;
    font-size: 1.8rem !important; /* Clean, oversized numbers */
    margin-top: 0.5rem !important;
    margin-bottom: 0 !important;
}

/* Sub-text information */
.desc-box {
    font-size: 0.8rem !important;
    color: #7c7c80 !important; 
    margin-top: -0.5rem !important;
    margin-bottom: 1rem !important;
    font-style: normal !important;
    letter-spacing: 0.02em;
}

/* Style default file uploader box */
div[data-testid="stFileUploader"] {
    border: 1px dashed #48484a !important;
    background-color: #0d0d0d !important;
    padding: 1.5rem !important;
    border-radius: 4px;
    transition: border-color 0.2s ease;
}

div[data-testid="stFileUploader"]:hover {
    border-color: #c5a880 !important;
}

/* =======================================================
   MINIMALIST SLIDER OVERRIDES
   ======================================================= */

/* The track bar container text */
div[data-testid="stSlider"] [data-testid="stSliderTickBar"] {
    color: #7c7c80 !important;
    font-size: 0.75rem;
}

/* Target baseline tracks */
div[data-testid="stSlider"] .st-ae {
    background-color: #2c2c2e !important; 
    height: 4px !important; /* Sleeker, thinner line */
    border: none !important;
}

/* Target active/filled track selection */
div[data-testid="stSlider"] .st-af {
    background-color: #c5a880 !important; /* Active fill matches gold accent */
}

/* The elegant circle draggable slider handle (Thumb) */
div[data-testid="stSlider"] [role="slider"] {
    background-color: #ffffff !important;
    border: 1px solid #ffffff !important;
    border-radius: 50% !important; /* Smooth circular button */
    width: 16px !important;
    height: 16px !important;
    box-shadow: 0 2px 6px rgba(0,0,0,0.4) !important;
    transition: transform 0.2s ease, background-color 0.2s !important;
}

/* Hover state for slider block */
div[data-testid="stSlider"] [role="slider"]:hover {
    transform: scale(1.2);
    background-color: #c5a880 !important;
    border-color: #c5a880 !important;
}

/* Max/Min bounds value texts */
div[data-testid="stSlider"] [data-testid="stWidgetLabel"] + div div {
    color: #aeaea3 !important;
    font-family: inherit !important;
    font-weight: 400 !important;
    font-size: 0.85rem;
}
""", unsafe_allow_html=True)

# Layout Setup
st.markdown('<div class="app-title"> image morpher </div>', unsafe_allow_html=True)
st.markdown('<div class="app-subtitle">upload your picture to transform it into a son meme</div>', unsafe_allow_html=True)

# Target Image Configuration
TARGET_PATH = "picture/saturn.jpg"

if not os.path.exists(TARGET_PATH):
    st.error(f"Target image not found at '{TARGET_PATH}'! verify the folder structure.")
    st.stop()
    
# Load Target Image (in background only, user requested not to show it)
target_image = Image.open(TARGET_PATH)

# ==================== SIDEBAR SETTINGS ====================
st.sidebar.markdown('<div class="sidebar-header">Controls</div>', unsafe_allow_html=True)

algorithm = st.sidebar.selectbox(
    "Choose Transformation Strategy",
    ["Neural Morphing (INR MLP)", "Neural Style Transfer (VGG-19)", "Fast Color Transfer (Reinhard)"],
    help="Select the AI or statistical method to transform the image."
)

resolution = st.sidebar.select_slider(
    "Target Image Sharpness",
    options=[128, 256, 512],
    value=256,
    help="Determines the size of the canvas. Higher is sharper but takes longer for the AI to compute."
)
st.sidebar.markdown('<div class="desc-box">Higher resolutions are sharper but will process slower on CPU.</div>', unsafe_allow_html=True)

st.sidebar.markdown('<div class="sidebar-section-title">AI Settings</div>', unsafe_allow_html=True)

# Conditional hyperparameters with layman explanations
if algorithm == "Neural Morphing (INR MLP)":
    # 1. Fit original image epochs
    fit_epochs = st.sidebar.slider(
        "Input Image Memorization",
        20, 150, 60, 10,
        help="How many rounds the AI spends studying your uploaded picture. Higher values help the AI remember your original photo more clearly before beginning the morph."
    )
    st.sidebar.markdown('<div class="desc-box">Determines how long the AI studies your photo to create its starting shape.</div>', unsafe_allow_html=True)
    
    # 2. Morphing epochs
    morph_epochs = st.sidebar.slider(
        "Morphing Duration",
        20, 200, 80, 10,
        help="The amount of time the AI spends shifting its focus from your picture to the target picture. Higher values produce a slower, more detailed transition."
    )
    st.sidebar.markdown('<div class="desc-box">Controls the duration of the transition phase.</div>', unsafe_allow_html=True)
    
    # 3. Learning rate / AI speed
    lr = st.sidebar.slider(
        "AI Learning Speed",
        0.001, 0.05, 0.01, 0.001,
        format="%.3f",
        help="How fast the AI adjusts its painting at each step. Setting this too fast may result in a chaotic, messy image; setting it too slow will prevent it from morphing fully."
    )
    st.sidebar.markdown('<div class="desc-box">How quickly the AI adapts. Recommend 0.010 for balanced results.</div>', unsafe_allow_html=True)
    
elif algorithm == "Neural Style Transfer (VGG-19)":
    # 1. Steps
    iterations = st.sidebar.slider(
        "Styling Duration",
        10, 150, 50, 5,
        help="The number of painting strokes/cycles the AI applies to match the textures of the target image. Higher values lead to a more heavily stylized painting."
    )
    st.sidebar.markdown('<div class="desc-box">Cycles of repainting the image. More cycles = heavier texture.</div>', unsafe_allow_html=True)
    
    # 2. Style Weight
    style_weight = st.sidebar.select_slider(
        "Texture Intensity",
        options=[1e3, 1e4, 1e5, 1e6, 1e7],
        value=1e5,
        help="How aggressively the texture and colors of the target image are forced onto your photo. Higher settings make the target details dominant."
    )
    st.sidebar.markdown('<div class="desc-box">Intensity of details/brushstrokes applied from target.</div>', unsafe_allow_html=True)
    
    # 3. Content Weight
    content_weight = st.sidebar.slider(
        "Outline Preservation",
        0.1, 10.0, 1.0, 0.1,
        help="How strongly the AI preserves the structure and shapes of your original photo. Higher numbers prevent your photo from dissolving too much into the style."
    )
    st.sidebar.markdown('<div class="desc-box">Preservation of original outlines and structure.</div>', unsafe_allow_html=True)
    
    # 4. Learning rate / Paint speed
    lr = st.sidebar.slider(
        "Painting Speed",
        0.01, 0.1, 0.03, 0.01,
        help="How large the visual modifications are at each step. High speed is faster but might overlook subtle details; lower speed is slower but precise."
    )
    st.sidebar.markdown('<div class="desc-box">How fast the canvas colors change.</div>', unsafe_allow_html=True)

elif algorithm == "Fast Color Transfer (Reinhard)":
    st.sidebar.info("Instant Transfer: This statistical method matches the lighting and colors immediately, requiring no iterative training cycles.")


# ==================== MAIN PANEL ====================

# Single Column Layout as requested
# 1. Big Upload Box Card
st.markdown('<div class="upload-card">', unsafe_allow_html=True)
st.markdown('<div class="upload-card-title">Upload Your Image</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

source_image = None
if uploaded_file is not None:
    source_image = Image.open(uploaded_file)
    st.image(source_image, caption="Your Original Image", use_container_width=True)
else:
    st.info("Upload a picture to run the neural transmutation!")
    # Use a default fallback colored pattern to allow tests
    placeholder_w, placeholder_h = 256, 256
    grad = Image.new("RGB", (placeholder_w, placeholder_h))
    pixels = grad.load()
    for x in range(placeholder_w):
        for y in range(placeholder_h):
            pixels[x, y] = (int(x / placeholder_w * 255), 100, int(y / placeholder_h * 255))
    source_image = grad
    st.image(source_image, caption="Default Placeholder Image (Upload a file to override)", use_container_width=True)
    
st.markdown('</div>', unsafe_allow_html=True)

# 2. Trigger Button (centered in flow)
run_btn = st.button("Start Transmutation")

# 3. Output Card Below Upload Box
if run_btn:
    if uploaded_file is None:
        st.warning("Using default placeholder image. Upload a picture above to transmute your own photo!")
        
    st.markdown('<div class="morphed-card">', unsafe_allow_html=True)
    st.markdown('<div class="morphed-card-title">Transmuted Output</div>', unsafe_allow_html=True)
    
    # Placeholders inside the output card
    status_box = st.empty()
    progress_bar = st.progress(0)
    
    col_out1, col_out2 = st.columns([1, 2])
    
    with col_out1:
        metrics_box = st.empty()
        download_box = st.empty()
        
    with col_out2:
        image_box = st.empty()
        
    result_image = None
    
    try:
        # Run specific algorithm
        if algorithm == "Neural Morphing (INR MLP)":
            total_epochs = fit_epochs + morph_epochs
            
            generator = run_neural_morph(
                source_img=source_image,
                target_img=target_image,
                resolution=resolution,
                fit_epochs=fit_epochs,
                morph_epochs=morph_epochs,
                lr=lr
            )
            
            for current_img, epoch, loss, phase in generator:
                # Update progress
                progress_val = float(epoch + 1) / total_epochs
                progress_bar.progress(progress_val)
                
                status_box.success(f"🤖 AI Action: **{phase}**")
                
                # Simplified training progress metrics
                metrics_box.markdown(f"""
                <div class="metrics-container">
                    <h4>AI Learning Stats</h4>
                    <p><b>Phase:</b> {phase}</p>
                    <p><b>Step:</b> {epoch + 1} / {total_epochs}</p>
                    <p><b>Error Margin:</b> <code style='color:#000; font-weight:800;'>{loss:.5f}</code></p>
                </div>
                """, unsafe_allow_html=True)
                
                # Display output image live
                image_box.image(current_img, caption=f"Morphed Canvas (Step {epoch+1})", use_container_width=True)
                result_image = current_img
                
            status_box.success("Transmutation Complete! The neural network has successfully morphed the image.")
            
        elif algorithm == "Neural Style Transfer (VGG-19)":
            generator = run_style_transfer(
                content_img=source_image,
                style_img=target_image,
                resolution=resolution,
                iterations=iterations,
                lr=lr,
                content_weight=content_weight,
                style_weight=style_weight
            )
            
            for current_img, step, loss, phase in generator:
                # Update progress
                progress_val = float(step) / iterations
                progress_bar.progress(progress_val)
                
                status_box.success(f"AI Action: **{phase}**")
                
                # Simplified metrics
                metrics_box.markdown(f"""
                <div class="metrics-container">
                    <h4>Optimization Stats</h4>
                    <p><b>Step:</b> {step} / {iterations}</p>
                    <p><b>Texture Error:</b> <code style='color:#000; font-weight:800;'>{loss:.1f}</code></p>
                </div>
                """, unsafe_allow_html=True)
                
                # Display output image live
                image_box.image(current_img, caption=f"Style Transfer Canvas (Step {step})", use_container_width=True)
                result_image = current_img
                
            status_box.success("Transmutation Complete! The VGG style network has finished painting the canvas.")
            
        elif algorithm == "Fast Color Transfer (Reinhard)":
            status_box.success(" Computing color distribution maps...")
            progress_bar.progress(0.5)
            time.sleep(0.3)
            
            result_image = transfer_color(source_image, target_image)
            
            progress_bar.progress(1.0)
            
            metrics_box.markdown("""
            <div class="metrics-container" style="background-color: #86efac !important;">
                <h4>Instant Statistics</h4>
                <p><b>Method:</b> Color Mapping</p>
                <p><b>Execution Time:</b> <code style='color:#000; font-weight:800;'>&lt; 0.05s</code></p>
            </div>
            """, unsafe_allow_html=True)
            
            image_box.image(result_image, caption="Color Mapped Output", use_container_width=True)
            status_box.success("Transmutation Complete! Target colors successfully mapped.")
            
        # Download utilities
        if result_image is not None:
            buf = io.BytesIO()
            result_image.save(buf, format="PNG")
            byte_im = buf.getvalue()
            
            with download_box:
                st.download_button(
                    label="Download Morph Output",
                    data=byte_im,
                    file_name="morphed_image.png",
                    mime="image/png"
                )
                
    except Exception as err:
        st.error(f"Transmutation failed: {err}")
        st.info("Check if your system has an internet connection (required on first run to download standard model weights).")
        
    st.markdown('</div>', unsafe_allow_html=True)
