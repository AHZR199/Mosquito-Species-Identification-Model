import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.models import resnet50
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def load_model(model_path):
    model = resnet50(weights=None)  # Updated to use weights argument
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 70)  # Update the number of classes as needed
    state_dict = torch.load(model_path)
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model.eval()
    return model

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((480, 640)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor

def grad_cam(model, image_tensor, target_layer):
    feature_maps = None
    gradients = None

    def forward_hook(module, input, output):
        nonlocal feature_maps
        feature_maps = output

    def backward_hook(module, grad_in, grad_out):
        nonlocal gradients
        gradients = grad_out[0]

    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)

    output = model(image_tensor)
    pred_idx = output.argmax(dim=1).item()

    model.zero_grad()
    one_hot_output = torch.FloatTensor(1, output.size()[-1]).zero_()
    one_hot_output[0][pred_idx] = 1
    output.backward(gradient=one_hot_output)

    weights = torch.mean(gradients, [2, 3], keepdim=True)
    cam = torch.sum(weights * feature_maps, dim=1, keepdim=True)
    cam = F.relu(cam)
    cam = cam - cam.min()
    cam = cam / cam.max()

    forward_handle.remove()
    backward_handle.remove()

    return cam.squeeze().cpu().detach().numpy()

def visualize_grad_cam(model, image_tensor, image_path, layers_to_visualize):
    image = Image.open(image_path)
    image_dir = os.path.dirname(image_path)

    for layer_name in layers_to_visualize:
        layer = dict([*model.named_modules()])[layer_name]
        cam = grad_cam(model, image_tensor, layer)
        heatmap = np.uint8(255 * cm.jet(cam)[..., :3])
        heatmap = Image.fromarray(heatmap).resize((image.width, image.height))
        blended = Image.blend(image, heatmap, alpha=0.5)
        
        output_path = os.path.join(image_dir, f"{layer_name}_grad_cam.png")
        blended.save(output_path)
        print(f"Saved: {output_path}")

# Usage example
model_path = "/work/soghigian_lab/abdullah.zubair/rerun4THE_BEST_MODEL.pth"
image_path = "/work/soghigian_lab/abdullah.zubair/523/IMG_3249.jpeg"
layers_to_visualize = ["layer1", "layer2", "layer3", "layer4"]

model = load_model(model_path)
image_tensor = preprocess_image(image_path)
visualize_grad_cam(model, image_tensor, image_path, layers_to_visualize)
