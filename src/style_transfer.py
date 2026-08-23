import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

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
            '21': 'conv4_2', 
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
    features = tensor.view(d, h * w)
    gram = torch.mm(features, features.t())
    return gram.div(d * h * w)

def run_style_transfer(content_img: Image.Image, style_img: Image.Image, 
                       resolution=256, iterations=50, lr=0.03,
                       content_weight=1.0, style_weight=1e5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   
    h, w = resolution, resolution
    
    loader = transforms.Compose([
        transforms.Resize((h, w)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN.tolist(), std=IMAGENET_STD.tolist())
    ])
    
    content_tensor = loader(content_img.convert("RGB")).unsqueeze(0).to(device)
    style_tensor = loader(style_img.convert("RGB")).unsqueeze(0).to(device)
    
    # Load vgg-19 model features (pretrained)
    try:
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.to(device).eval()
    except Exception as e:
        # fallback to local files or raise custom error if no internet connection
        raise RuntimeError(f"Failed to load VGG-19 weights: {e}. Please ensure you have internet access for the first execution.")
        
    for param in vgg.parameters():
        param.requires_grad = False

    with torch.no_grad():
        content_features = get_vgg_features(content_tensor, vgg)
        style_features = get_vgg_features(style_tensor, vgg)
        style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}
        target_tensor = content_tensor.clone().requires_grad_(True)
        optimizer = optim.Adam([target_tensor], lr=lr)
        style_layer_weights = {
            'conv1_1': 1.0,
            'conv2_1': 0.8,
            'conv3_1': 0.5,
            'conv4_1': 0.3,
            'conv5_1': 0.1
    }
   
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
        
    for i in range(1, iterations + 1):
        target_features = get_vgg_features(target_tensor, vgg)
        content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2']) ** 2)
        style_loss = 0
        for layer in style_layer_weights:
            target_feature = target_features[layer]
            target_gram = gram_matrix(target_feature)
            style_gram = style_grams[layer]
            layer_style_loss = style_layer_weights[layer] * torch.mean((target_gram - style_gram) ** 2)
            style_loss += layer_style_loss
            total_loss = content_weight * content_loss + style_weight * style_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        if i % 2 == 0 or i == iterations:
            yield get_pil_image(target_tensor), i, total_loss.item(), "Styling Canvas..."
