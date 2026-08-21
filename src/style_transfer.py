import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# ImageNet normalization parameters
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])

def get_vgg_features(image, model, layers=None):
    """Extract features from specific layers of VGG-19."""
    if layers is None:
        layers = {
            '0': 'conv1_1',
            '5': 'conv2_1',
            '10': 'conv3_1',
            '19': 'conv4_1',
            '21': 'conv4_2',  # Content layer
            '28': 'conv5_1'
        }
    
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
            
    return features

def gram_matrix(tensor):
    """Calculate the Gram Matrix of a given tensor."""
    _, d, h, w = tensor.size()
    # Flatten spatial dimensions
    features = tensor.view(d, h * w)
    # Calculate outer product
    gram = torch.mm(features, features.t())
    # Normalize by total number of elements in feature map
    return gram.div(d * h * w)

def run_style_transfer(content_img: Image.Image, style_img: Image.Image, 
                       resolution=256, iterations=50, lr=0.03,
                       content_weight=1.0, style_weight=1e5):
    """
    Executes Neural Style Transfer on CPU/GPU.
    Yields: (current_pil_image, iteration_num, loss_value, status_string)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Define transformations: resize, convert to tensor, normalize
    # Keep resolution modest on CPU to prevent excessive wait times
    h, w = resolution, resolution
    
    loader = transforms.Compose([
        transforms.Resize((h, w)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN.tolist(), std=IMAGENET_STD.tolist())
    ])
    
    # Load images
    content_tensor = loader(content_img).unsqueeze(0).to(device)
    style_tensor = loader(style_img).unsqueeze(0).to(device)
    
    # Load VGG-19 model features (pretrained)
    # This might take a bit on first load to download (~500MB)
    try:
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.to(device).eval()
    except Exception as e:
        # Fallback to local files or raise custom error if no internet connection
        raise RuntimeError(f"Failed to load VGG-19 weights: {e}. Please ensure you have internet access for the first execution.")
        
    # Freeze model weights
    for param in vgg.parameters():
        param.requires_grad = False
        
    # Get initial features
    content_features = get_vgg_features(content_tensor, vgg)
    style_features = get_vgg_features(style_tensor, vgg)
    
    # Compute style Gram matrices
    style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}
    
    # Initialize the target image as a clone of the content image (with gradients enabled)
    target_tensor = content_tensor.clone().requires_grad_(True)
    
    # Define optimizer
    optimizer = optim.Adam([target_tensor], lr=lr)
    
    # Style weights per layer
    style_layer_weights = {
        'conv1_1': 1.0,
        'conv2_1': 0.8,
        'conv3_1': 0.5,
        'conv4_1': 0.3,
        'conv5_1': 0.1
    }
    
    # Inverse transform to get back to displayable PIL image
    denormalize = transforms.Normalize(
        mean=(-IMAGENET_MEAN / IMAGENET_STD).tolist(),
        std=(1.0 / IMAGENET_STD).tolist()
    )
    
    def get_pil_image(tensor):
        t = tensor.clone().detach().squeeze(0)
        t = denormalize(t)
        t = torch.clamp(t, 0.0, 1.0)
        t = t.permute(1, 2, 0).cpu().numpy()
        return Image.fromarray((t * 255.0).astype(np.uint8))
        
    # Optimization loop
    for i in range(1, iterations + 1):
        # Extract features for current target
        target_features = get_vgg_features(target_tensor, vgg)
        
        # Calculate Content Loss
        content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2']) ** 2)
        
        # Calculate Style Loss
        style_loss = 0
        for layer in style_layer_weights:
            target_feature = target_features[layer]
            target_gram = gram_matrix(target_feature)
            style_gram = style_grams[layer]
            
            layer_style_loss = style_layer_weights[layer] * torch.mean((target_gram - style_gram) ** 2)
            style_loss += layer_style_loss
            
        # Total loss
        total_loss = content_weight * content_loss + style_weight * style_loss
        
        # Optimization step
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # Yield intermediate result
        if i % 2 == 0 or i == iterations:
            yield get_pil_image(target_tensor), i, total_loss.item(), "Styling Canvas..."
