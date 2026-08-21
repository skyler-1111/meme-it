import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PIL import Image

class PositionalEncoding(nn.Module):
    def __init__(self, num_frequencies=6):
        super(PositionalEncoding, self).__init__()
        self.num_frequencies = num_frequencies
        # Frequency bands: 2^0, 2^1, ... 2^(num_frequencies-1)
        self.frequencies = torch.tensor([2.0 ** i for i in range(num_frequencies)], dtype=torch.float32)

    def forward(self, coords):
        # coords shape: [N, 2]
        device = coords.device
        freqs = self.frequencies.to(device)
        
        # Multiply coordinates by frequency bands
        # Shape: [N, 2, num_frequencies]
        args = coords.unsqueeze(-1) * freqs * np.pi
        
        # Calculate sine and cosine
        sin_vals = torch.sin(args)
        cos_vals = torch.cos(args)
        
        # Flatten frequencies and concatenate with raw coordinates
        # Out shape: [N, 2 + 2 * 2 * num_frequencies] -> [N, 2 + 4 * num_frequencies]
        features = torch.cat([coords, sin_vals.flatten(start_dim=1), cos_vals.flatten(start_dim=1)], dim=-1)
        return features

class CoordinateMLP(nn.Module):
    def __init__(self, in_features, hidden_dim=128, num_layers=3):
        super(CoordinateMLP, self).__init__()
        layers = []
        
        # Input layer
        layers.append(nn.Linear(in_features, hidden_dim))
        layers.append(nn.ReLU())
        
        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            
        # Output layer (RGB values between 0 and 1)
        layers.append(nn.Linear(hidden_dim, 3))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

def get_coordinate_grid(h, w, device):
    """Generates coordinate grid normalized from -1 to 1."""
    y = torch.linspace(-1, 1, h, device=device)
    x = torch.linspace(-1, 1, w, device=device)
    grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
    # Shape: [H * W, 2]
    coords = torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2)
    return coords

def image_to_tensor(img, h, w, device):
    """Converts a PIL Image to a flat RGB tensor [H*W, 3] on target device."""
    img_resized = img.resize((w, h), Image.Resampling.LANCZOS)
    arr = np.array(img_resized).astype(np.float32) / 255.0
    tensor = torch.tensor(arr, device=device).reshape(-1, 3)
    return tensor

def tensor_to_image(tensor, h, w):
    """Converts a flat RGB tensor [H*W, 3] back to a PIL Image."""
    arr = tensor.detach().cpu().numpy().reshape(h, w, 3)
    arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
    return Image.fromarray(arr)

def run_neural_morph(source_img: Image.Image, target_img: Image.Image, 
                     resolution=128, fit_epochs=60, morph_epochs=80, lr=0.01):
    """
    Trains a coordinate MLP to represent the source image, then morphs it into the target image.
    Yields: (current_pil_image, epoch_num, loss_value, phase_string)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # We maintain the aspect ratio of the target image (imcrine.jpg) or keep it square.
    # Let's resize images to a standard resolution.
    # We will use width = resolution, height = resolution for simple square grid
    h, w = resolution, resolution
    
    # Convert images to tensors
    source_tensor = image_to_tensor(source_img, h, w, device)
    target_tensor = image_to_tensor(target_img, h, w, device)
    
    # Prepare inputs and model
    coords = get_coordinate_grid(h, w, device)
    pos_encoder = PositionalEncoding(num_frequencies=6)
    encoded_coords = pos_encoder(coords)
    
    in_features = encoded_coords.shape[-1]
    model = CoordinateMLP(in_features=in_features, hidden_dim=128, num_layers=3).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    # Phase 1: Fit the Source Image
    for epoch in range(fit_epochs):
        optimizer.zero_grad()
        predictions = model(encoded_coords)
        loss = criterion(predictions, source_tensor)
        loss.backward()
        optimizer.step()
        
        # Yield progress occasionally
        if epoch % 5 == 0 or epoch == fit_epochs - 1:
            current_img = tensor_to_image(predictions, h, w)
            yield current_img, epoch, loss.item(), "Fitting Uploaded Image"
            
    # Phase 2: Morphing to Target Image (imcrine.jpg)
    for epoch in range(morph_epochs):
        alpha = epoch / float(morph_epochs - 1) if morph_epochs > 1 else 1.0
        
        # Target linearly interpolated in pixel space for training guidance
        current_target = (1.0 - alpha) * source_tensor + alpha * target_tensor
        
        optimizer.zero_grad()
        predictions = model(encoded_coords)
        loss = criterion(predictions, current_target)
        loss.backward()
        optimizer.step()
        
        # Yield progress at each step (or every 2 steps to avoid UI lag)
        if epoch % 2 == 0 or epoch == morph_epochs - 1:
            current_img = tensor_to_image(predictions, h, w)
            yield current_img, fit_epochs + epoch, loss.item(), "Neural Morphing in Progress"
