# utils/visualization.py
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable


def grad_cam(model, image, target_layer):
    image = Variable(image, requires_grad=True)
    model_output = model(image)

    # Get the gradient of the target layer
    model_output.backward()
    gradient = image.grad.data

    # Get the weights from the gradients
    weights = torch.mean(gradient, dim=(2, 3), keepdim=True)

    # Weighted sum of the feature maps
    cam = torch.sum(weights * image, dim=1, keepdim=True)
    cam = torch.relu(cam)
    cam = cam / torch.max(cam)  # Normalize CAM

    return cam


def visualize_cam(cam, image):
    cam = cam.squeeze().cpu().numpy()
    cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # Normalize
    plt.imshow(image.permute(1, 2, 0).cpu().numpy())
    plt.imshow(cam, cmap='jet', alpha=0.5)  # Overlay CAM
    plt.show()
