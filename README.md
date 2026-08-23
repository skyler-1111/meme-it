**Image-Morpher**
<br>
An image transformation playground built with Python, Streamlit, PyTorch, and OpenCV-style image processing.

Upload an image, pick how you want it transformed, and let the machine cook.
-------------------------------------
**What it does**

img morpher currently has three transformation modes:

**1. Neural Morphing — INR MLP**

Uses a coordinate-based neural network (Implicit Neural Representation) to first memorize the uploaded image and then gradually morph it toward the target image.

Positional encoding
Coordinate-based MLP
PyTorch + Adam optimization
Adjustable fitting and morphing steps
CPU/GPU support

**2. Neural Style Transfer — VGG-19**

Uses a pretrained VGG-19 network to transfer the visual style of the target image onto the uploaded image.

Content loss
Style loss
Gram matrices
Configurable style/content weights
Adjustable optimization speed and iterations

The first run may need to download the VGG-19 weights.

**3. Fast Color Transfer — Reinhard**

The instant option.
Instead of training a neural network, it matches the color statistics of the uploaded image to the target image using Reinhard-style color transfer in Lαβ color space.

(sounds so tuff)
-------------------------------------

**Installation:**
<br>
You'll need Python 3.10+ and preferably a virtual environment.
<br>
1. Clone the repository
<br> `git clone https://github.com/E/img-morpher.git`
<br> `cd img-morpher`

<br> 3. Create a virtual environment

- Windows:
<br> `python -m venv venv`
<br> `venv\Scripts\activate`

- macOS/Linux:
<br> `python3 -m venv venv`
<br> `source venv/bin/activate`
 
- Install dependencies
<br> `pip install -r requirements.txt`
- Run the app
<br> `streamlit run app.py`

