import numpy as np
from PIL import Image

def rgb_to_lms(rgb):
    """onverts an RGB image [0, 255] to LMS color space."""
    # Scale to [0, 1]
    rgb = rgb.astype(np.float64) / 255.0
    # Avoid zero values before log
    rgb = np.clip(rgb, 1e-8, 1.0)
    
    # Transformation matrix
    matrix = np.array([
        [0.3811, 0.5783, 0.0402],
        [0.1967, 0.7244, 0.0782],
        [0.0241, 0.1288, 0.8444]
    ])
    
    # Reshape for matrix multiplication
    h, w, c = rgb.shape
    rgb_flat = rgb.reshape(-1, c)
    lms_flat = np.dot(rgb_flat, matrix.T)
    lms = lms_flat.reshape(h, w, c)
    return lms

def lms_to_rgb(lms):
    """Converts an LMS image to RGB [0, 255]."""
    matrix_inv = np.array([
        [4.4679, -3.5873, 0.1193],
        [-1.2186, 2.3809, -0.1624],
        [0.0497, -0.2439, 1.2045]
    ])
    
    h, w, c = lms.shape
    lms_flat = lms.reshape(-1, c)
    rgb_flat = np.dot(lms_flat, matrix_inv.T)
    rgb = rgb_flat.reshape(h, w, c)
    
    # Clip and scale to [0, 255]
    rgb = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    return rgb

def rgb_to_lab(rgb):
    """Converts RGB to orthogonal lab (l alpha beta) space."""
    lms = rgb_to_lms(rgb)
    lms_log = np.log10(lms)
    
    # Conversion to l-alpha-beta
    h, w, c = lms_log.shape
    l = (lms_log[:, :, 0] + lms_log[:, :, 1] + lms_log[:, :, 2]) / np.sqrt(3)
    a = (lms_log[:, :, 0] + lms_log[:, :, 1] - 2 * lms_log[:, :, 2]) / np.sqrt(6)
    b = (lms_log[:, :, 0] - lms_log[:, :, 1]) / np.sqrt(2)
    
    return np.stack([l, a, b], axis=-1)

def lab_to_rgb(lab):
    """Converts orthogonal lab (l alpha beta) space to RGB."""
    l = lab[:, :, 0]
    a = lab[:, :, 1]
    b = lab[:, :, 2]
    
    # Conversion back to LMS log space
    lms_l = (l / np.sqrt(3)) + (a / np.sqrt(6)) + (b / np.sqrt(2))
    lms_m = (l / np.sqrt(3)) + (a / np.sqrt(6)) - (b / np.sqrt(2))
    lms_s = (l / np.sqrt(3)) - (2 * a / np.sqrt(6))
    
    lms_log = np.stack([lms_l, lms_m, lms_s], axis=-1)
    lms = 10 ** lms_log
    return lms_to_rgb(lms)

def transfer_color(source_img: Image.Image, target_img: Image.Image) -> Image.Image:
    """
    Transforms the color of the source image to match the target image.
    Uses Reinhard's color transfer in the l-alpha-beta space.
    """
    # Resize target to match source for pixel alignment if necessary, 
    # but statistics don't strictly require matching dimensions.
    # Convert PIL Images to numpy arrays
    source_arr = np.array(source_img.convert("RGB"))
    target_arr = np.array(target_img.convert("RGB"))
    
    # Convert to lab color space
    source_lab = rgb_to_lab(source_arr)
    target_lab = rgb_to_lab(target_arr)
    
    # Compute mean and std for source
    source_mean = np.mean(source_lab, axis=(0, 1))
    source_std = np.std(source_lab, axis=(0, 1))
    
    # Compute mean and std for target
    target_mean = np.mean(target_lab, axis=(0, 1))
    target_std = np.std(target_lab, axis=(0, 1))
    
    # Avoid division by zero
    source_std = np.where(source_std == 0, 1e-5, source_std)
    
    # Perform color transfer channel-by-channel
    result_lab = np.zeros_like(source_lab)
    for channel in range(3):
        # Scale and shift channel: (val - mean_s) * (std_t / std_s) + mean_t
        result_lab[:, :, channel] = (
            (source_lab[:, :, channel] - source_mean[channel]) * 
            (target_std[channel] / source_std[channel]) + 
            target_mean[channel]
        )
        
    # Convert back to RGB
    result_arr = lab_to_rgb(result_lab)
    return Image.fromarray(result_arr)
