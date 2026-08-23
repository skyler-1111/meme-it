import streamlit as st
import os
from PIL import Image
import time
import io

# import custom image processing algorithms
from src.color_transfer import transfer_color
from src.neural_morph import run_neural_morph
from src.style_transfer import run_style_transfer

st.set_page_config(
    page_title="img morpher",
    page_icon="🗿",
    layout="centered", 
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.stApp {
    background-color: #0d0d0d;
    color: #f5f5f7;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    letter-spacing: -0.02em;
}

.app-title {
    text-align: center;
    background-color: #1a1a1a; 
    color: #ffffff;
    border: 1px solid #2d2d2d;
    box-shadow: 0px 8px 30px rgba(0, 0, 0, 0.5); 
    padding: 2rem;
    font-size: 2.5rem;
    font-weight: 300;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-top: 1.5rem;
    margin-bottom: 0.5rem;
}

.app-subtitle {
    text-align: center;
    color: #c5a880; 
    font-size: 0.9rem;
    font-weight: 500;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 2.5rem;
    background-color: transparent; 
    border: none;
    box-shadow: none;
    padding: 0;
}

section[data-testid="stSidebar"] {
    background-color: #050505; 
    border-right: 1px solid #1c1c1e;
}

section[data-testid="stSidebar"] * {
    color: #e5e5ea;
}

div[data-testid="stWidgetLabel"] p {
    font-weight: 500;
    font-size: 0.85rem;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    color: #a1a1aa;
}

.sidebar-header {
    font-size: 1.1rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    background-color: transparent; 
    color: #ffffff;
    padding: 0.5rem 0;
    border-bottom: 1px solid #2d2d2d;
    margin-bottom: 1.5rem;
}

.sidebar-section-title {
    font-size: 0.85rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    border-bottom: 1px solid #2d2d2d;
    margin-top: 2rem;
    margin-bottom: 1rem;
    padding-bottom: 0.4rem;
    color: #c5a880; 
}

.upload-card {
    background-color: #161617;
    border: 1px solid #2c2c2e;
    padding: 2rem;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    margin-bottom: 2rem;
    border-radius: 4px;
}

.upload-card-title {
    font-size: 1rem;
    font-weight: 600;
    color: #ffffff;
    background-color: transparent; 
    border: none;
    padding: 0;
    margin-bottom: 1.2rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    display: block;
    box-shadow: none;
}

.stButton>button {
    background-color: #ffffff; 
    color: #000000;
    border: 1px solid #ffffff;
    border-radius: 2px; 
    padding: 0.8rem 2rem;
    font-weight: 600;
    font-size: 1rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    box-shadow: none;
    transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
    width: 100%;
    margin-bottom: 2rem;
}

.stButton>button:hover {
    background-color: transparent;
    color: #ffffff;
    border-color: #ffffff;
    cursor: pointer;
}

.stButton>button:active {
    transform: scale(0.98);
    background-color: rgba(255, 255, 255, 0.1);
}

.morphed-card {
    background-color: #161617;
    border: 1px solid #2c2c2e;
    padding: 2rem;
    box-shadow: 0 10px 30px rgba(0,0,0,0.4);
    margin-top: 1.5rem;
    margin-bottom: 2rem;
    border-radius: 4px;
}

.morphed-card-title {
    font-size: 1rem;
    font-weight: 600;
    color: #c5a880;
    background-color: transparent; 
    border: none;
    padding: 0;
    margin-bottom: 1.2rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    display: block;
    box-shadow: none;
}

.metrics-container {
    background-color: #1c1c1e;
    border: 1px solid #2c2c2e;
    padding: 1.2rem;
    box-shadow: none;
    color: #ffffff;
    margin-bottom: 1rem;
    border-radius: 4px;
}

.metrics-container h4 {
    color: #ffffff;
    font-weight: 500;
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-top: 0;
    border-bottom: 1px solid #2c2c2e;
    padding-bottom: 0.5rem;
}

.metrics-container p {
    color: #c5a880; 
    font-weight: 300;
    font-size: 1.8rem; 
    margin-top: 0.5rem;
    margin-bottom: 0;
}

.desc-box {
    font-size: 0.8rem;
    color: #7c7c80; 
    margin-top: -0.5rem;
    margin-bottom: 1rem;
    font-style: normal;
    letter-spacing: 0.02em;
}

div[data-testid="stFileUploader"] {
    border: 1px dashed #48484a;
    background-color: #0d0d0d;
    padding: 1.5rem;
    border-radius: 4px;
    transition: border-color 0.2s ease;
}

div[data-testid="stFileUploader"]:hover {
    border-color: #c5a880;
}

div[data-testid="stSlider"] [data-testid="stSliderTickBar"] {
    color: #7c7c80;
    font-size: 0.75rem;
}

div[data-testid="stSlider"] .st-ae {
    background-color: #2c2c2e; 
    height: 4px; /* Sleeker, thinner line
    border: none;
}

div[data-testid="stSlider"] .st-af {
    background-color: #c5a880; 
}

div[data-testid="stSlider"] [role="slider"] {
    background-color: #ffffff;
    border: 1px solid #ffffff;
    border-radius: 50%; 
    width: 16px;
    height: 16px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.4);
    transition: transform 0.2s ease, background-color 0.2s;
}

div[data-testid="stSlider"] [role="slider"]:hover {
    transform: scale(1.2);
    background-color: #c5a880;
    border-color: #c5a880;
}

div[data-testid="stSlider"] [data-testid="stWidgetLabel"] + div div {
    color: #aeaea3;
    font-family: inherit;
    font-weight: 400;
    font-size: 0.85rem;
}
""", unsafe_allow_html=True)

st.markdown('<div class="app-title"> image morpher </div>', unsafe_allow_html=True)
st.markdown('<div class="app-subtitle">upload your picture to transform it into a son meme</div>', unsafe_allow_html=True)

TARGET_PATH = "picture/imcrine.jpg"

if not os.path.exists(TARGET_PATH):
    st.error(f"Target image not found at '{TARGET_PATH}'! verify the folder structure.")
    st.stop()

target_image = Image.open(TARGET_PATH).convert("RGB")

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

if algorithm == "Neural Morphing (INR MLP)":
    fit_epochs = st.sidebar.slider(
        "Input Image Memorization",
        20, 150, 60, 10,
        help="How many rounds the AI spends studying your uploaded picture. Higher values help the AI remember your original photo more clearly before beginning the morph."
    )
    st.sidebar.markdown('<div class="desc-box">Determines how long the AI studies your photo to create its starting shape.</div>', unsafe_allow_html=True)

    morph_epochs = st.sidebar.slider(
        "Morphing Duration",
        20, 200, 80, 10,
        help="The amount of time the AI spends shifting its focus from your picture to the target picture. Higher values produce a slower, more detailed transition."
    )
    st.sidebar.markdown('<div class="desc-box">Controls the duration of the transition phase.</div>', unsafe_allow_html=True)
   
    lr = st.sidebar.slider(
        "AI Learning Speed",
        0.001, 0.05, 0.01, 0.001,
        format="%.3f",
        help="How fast the AI adjusts its painting at each step. Setting this too fast may result in a chaotic, messy image; setting it too slow will prevent it from morphing fully."
    )
    st.sidebar.markdown('<div class="desc-box">How quickly the AI adapts.</div>', unsafe_allow_html=True)
    
elif algorithm == "Neural Style Transfer (VGG-19)":
 
    iterations = st.sidebar.slider(
        "Styling Duration",
        10, 150, 50, 5,
        help="The number of painting strokes/cycles the AI applies to match the textures of the target image. Higher values lead to a more heavily stylized painting."
    )
    st.sidebar.markdown('<div class="desc-box">Cycles of repainting the image. More cycles = heavier texture.</div>', unsafe_allow_html=True)
  
    style_weight = st.sidebar.select_slider(
        "Texture Intensity",
        options=[1e3, 1e4, 1e5, 1e6, 1e7],
        value=1e5,
        help="How aggressively the texture and colors of the target image are forced onto your photo. Higher settings make the target details dominant."
    )
    st.sidebar.markdown('<div class="desc-box">Intensity of details/brushstrokes applied from target.</div>', unsafe_allow_html=True)
  
    content_weight = st.sidebar.slider(
        "Outline Preservation",
        0.1, 10.0, 1.0, 0.1,
        help="How strongly the AI preserves the structure and shapes of your original photo. Higher numbers prevent your photo from dissolving too much into the style."
    )
    st.sidebar.markdown('<div class="desc-box">Preservation of original outlines and structure.</div>', unsafe_allow_html=True)
  
    lr = st.sidebar.slider(
        "Painting Speed",
        0.01, 0.1, 0.03, 0.01,
        help="How large the visual modifications are at each step. High speed is faster but might overlook subtle details; lower speed is slower but precise."
    )
    st.sidebar.markdown('<div class="desc-box">How fast the canvas colors change.</div>', unsafe_allow_html=True)

elif algorithm == "Fast Color Transfer (Reinhard)":
    st.sidebar.info("Instant Transfer: This statistical method matches the lighting and colors immediately, requiring no iterative training cycles.")

st.markdown('<div class="upload-card">', unsafe_allow_html=True)
st.markdown('<div class="upload-card-title">Upload Your Image</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

source_image = None
if uploaded_file is not None:
    source_image = Image.open(uploaded_file).convert("RGB")
    st.image(source_image, caption="Your Original Image", use_container_width=True)
else:
    st.info("Upload a picture to run the neural transmutation!")
    placeholder_w, placeholder_h = 256, 256
    grad = Image.new("RGB", (placeholder_w, placeholder_h))
    pixels = grad.load()
    for x in range(placeholder_w):
        for y in range(placeholder_h):
            pixels[x, y] = (int(x / placeholder_w * 255), 100, int(y / placeholder_h * 255))
    source_image = grad
    st.image(source_image, caption="Default Placeholder Image (Upload a file to override)", use_container_width=True)
    
st.markdown('</div>', unsafe_allow_html=True)

run_btn = st.button("Start Transmutation")

if run_btn:
    if uploaded_file is None:
        st.warning("Using default placeholder image. Upload a picture above to transmute your own photo!")
        
    st.markdown('<div class="morphed-card">', unsafe_allow_html=True)
    st.markdown('<div class="morphed-card-title">Transmuted Output</div>', unsafe_allow_html=True)
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
        # run specific algorithm
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
                progress_val = float(epoch + 1) / total_epochs
                progress_bar.progress(progress_val)
                
                status_box.success(f"AI Action: **{phase}**")
           
                metrics_box.markdown(f"""
                <div class="metrics-container">
                    <h4>AI Learning Stats</h4>
                    <p><b>Phase:</b> {phase}</p>
                    <p><b>Step:</b> {epoch + 1} / {total_epochs}</p>
                    <p><b>Error Margin:</b> <code style='color:#000; font-weight:800;'>{loss:.5f}</code></p>
                </div>
                """, unsafe_allow_html=True)
   
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
                progress_val = float(step) / iterations
                progress_bar.progress(progress_val)
                
                status_box.success(f"AI Action: **{phase}**")
                metrics_box.markdown(f"""
                <div class="metrics-container">
                    <h4>Optimization Stats</h4>
                    <p><b>Step:</b> {step} / {iterations}</p>
                    <p><b>Texture Error:</b> <code style='color:#000; font-weight:800;'>{loss:.1f}</code></p>
                </div>
                """, unsafe_allow_html=True)
               
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
