import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PIL import Image

class PositionalEncoding(nn.Module):
    def __init__(self, num_frequencies=6):
        super(PositionalEncoding, self).__init__()
        self.num_frequencies = num_frequencies
        self.frequencies = torch.tensor([2.0 ** i for i in range(num_frequencies)], dtype=torch.float32)

    def forward(self, coords):
        device = coords.device
        freqs = self.frequencies.to(device)
        args = coords.unsqueeze(-1) * freqs * np.pi
        
        #calc of sin n cosine 
        sin_vals = torch.sin(args)
        cos_vals = torch.cos(args)
    
        features = torch.cat([coords, sin_vals.flatten(start_dim=1), cos_vals.flatten(start_dim=1)], dim=-1)
        return features

class CoordinateMLP(nn.Module):
    def __init__(self, in_features, hidden_dim=128, num_layers=3):
        super(CoordinateMLP, self).__init__()
        layers = []
        layers.append(nn.Linear(in_features, hidden_dim))
        layers.append(nn.ReLU())
     
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_dim, 3))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

def get_coordinate_grid(h, w, device):
    y = torch.linspace(-1, 1, h, device=device)
    x = torch.linspace(-1, 1, w, device=device)
    grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
    coords = torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2)
    return coords

def image_to_tensor(img, h, w, device):
    img_resized = img.convert("RGB").resize((w, h), Image.Resampling.LANCZOS)
    arr = np.array(img_resized).astype(np.float32) / 255.0
    tensor = torch.tensor(arr, device=device).reshape(-1, 3)
    return tensor

def tensor_to_image(tensor, h, w):
    arr = tensor.detach().cpu().numpy().reshape(h, w, 3)
    arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
    return Image.fromarray(arr)

def run_neural_morph(source_img: Image.Image, target_img: Image.Image, 
                     resolution=128, fit_epochs=60, morph_epochs=80, lr=0.01):
                         
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    h, w = resolution, resolution
    # convert images to tensors
    source_tensor = image_to_tensor(source_img, h, w, device)
    target_tensor = image_to_tensor(target_img, h, w, device)
    
    # prepare da inputs n model
    coords = get_coordinate_grid(h, w, device)
    pos_encoder = PositionalEncoding(num_frequencies=6)
    encoded_coords = pos_encoder(coords)
    
    in_features = encoded_coords.shape[-1]
    model = CoordinateMLP(in_features=in_features, hidden_dim=128, num_layers=3).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    for epoch in range(fit_epochs):
        optimizer.zero_grad()
        predictions = model(encoded_coords)
        loss = criterion(predictions, source_tensor)
        loss.backward()
        optimizer.step()
        
        if epoch % 5 == 0 or epoch == fit_epochs - 1:
            current_img = tensor_to_image(predictions, h, w)
            yield current_img, epoch, loss.item(), "Fitting Uploaded Image"
            
    for epoch in range(morph_epochs):
        alpha = epoch / float(morph_epochs - 1) if morph_epochs > 1 else 1.0

        current_target = (1.0 - alpha) * source_tensor + alpha * target_tensor
        
        optimizer.zero_grad()
        predictions = model(encoded_coords)
        loss = criterion(predictions, current_target)
        loss.backward()
        optimizer.step()
   
        if epoch % 2 == 0 or epoch == morph_epochs - 1:
            current_img = tensor_to_image(predictions, h, w)
            yield current_img, fit_epochs + epoch, loss.item(), "Neural Morphing in Progress"
