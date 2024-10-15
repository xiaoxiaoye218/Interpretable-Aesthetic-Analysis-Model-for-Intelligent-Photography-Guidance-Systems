import torch
from PIL import Image
from torchvision import transforms
from models.cnn_feature_extractor import CNNFeatureExtractor
from models.attribute_score_predictor import AttributeScorePredictorWithAttention
from models.aesthetic_decomposition_network import AestheticDecompositionNetwork
import os
import json
import matplotlib.pyplot as plt
import numpy as np

# Load configuration
config_path = 'config.json'
if not os.path.exists(config_path):
    raise FileNotFoundError(f"Configuration file not found at {config_path}")

config = json.load(open(config_path))

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() and config['use_cuda'] else 'cpu')

# Initialize models and move them to the appropriate device
cnn = CNNFeatureExtractor().to(device)
attribute_predictor_with_attention = AttributeScorePredictorWithAttention(num_attributes=11).to(device)
decomposition_network = AestheticDecompositionNetwork(num_attributes=11).to(device)

# Load the trained model (specify the epoch you want to use, for example 50)
model_path = os.path.join(config['save_path'], f"epoch_{config['eval_epoch']}.pth")
checkpoint = torch.load(model_path, map_location=device)

cnn.load_state_dict(checkpoint['cnn_state_dict'])
attribute_predictor_with_attention.load_state_dict(checkpoint['attribute_predictor_state_dict'])
decomposition_network.load_state_dict(checkpoint['decomposition_network_state_dict'])

# Set models to evaluation mode
cnn.eval()
attribute_predictor_with_attention.eval()
decomposition_network.eval()

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Modify the size to match your model's input
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# Function to load and preprocess the image
def load_and_preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")  # Ensure 3 channels (RGB)
    return transform(image).unsqueeze(0)  # Add batch dimension


# Function to visualize the attention map
def visualize_attention(attention_map, original_image, attribute_name, score, weight, save_path):
    attention_map = attention_map.squeeze().cpu().detach().numpy()  # Remove batch and channel dimensions
    attention_map = (attention_map - attention_map.min()) / (
                attention_map.max() - attention_map.min())  # Normalize attention map
    attention_map = np.uint8(255 * attention_map)  # Convert to 8-bit for visualization

    # Resize attention map to match original image size
    attention_map = Image.fromarray(attention_map).resize(original_image.size, resample=Image.BILINEAR)

    # Convert attention map to RGBA (for transparency)
    attention_map = np.array(attention_map)
    attention_map_colored = plt.get_cmap('Blues')(attention_map / 255.0)  # Apply colormap
    attention_map_colored = np.uint8(255 * attention_map_colored[:, :, :3])  # Drop alpha channel

    # Convert original image to RGBA
    original_image_rgba = original_image.convert("RGBA")
    attention_overlay = Image.fromarray(attention_map_colored).convert("RGBA")

    # Blend the attention map with original image (use transparency to overlay)
    blended_image = Image.blend(original_image_rgba, attention_overlay, alpha=0.5)  # Adjust alpha for transparency

    # Plot the blended image
    plt.imshow(blended_image)
    plt.title(f"{attribute_name}\nWeight: {weight:.4f}, Score: {score:.4f}")
    plt.axis('off')  # Turn off axis for a clean image

    # Ensure save directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save the figure
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()


# Load and preprocess the image
image_path = 'data/images/farm1_255_19452343093_8ee7e5e375_b.jpg'  # Specify your image path here
image_tensor = load_and_preprocess_image(image_path).to(device)

# Load original image for visualization (before transformation)
original_image = Image.open(image_path).convert("RGB")

# Forward pass through the model
with torch.no_grad():
    features = cnn(image_tensor)
    attribute_scores, attention_maps = attribute_predictor_with_attention(features)  # Get scores and attention maps
    weights = decomposition_network(features)  # Predict attribute weights
    overall_score = torch.sum(attribute_scores * weights, dim=1).item()  # Calculate overall score

# Print results
attribute_names = ['BalancingElements', 'ColorHarmony', 'Content', 'DoF', 'Light', 'MotionBlur',
                   'Object', 'Repetition', 'RuleOfThirds', 'Symmetry', 'VividColor']

print("Predicted Attribute Scores:")
for i, score in enumerate(attribute_scores[0]):
    print(f"{attribute_names[i]}: {score.item():.4f}")

print("\nPredicted Attribute Weights:")
for i, weight in enumerate(weights[0]):
    print(f"{attribute_names[i]}: {weight.item():.4f}")

print(f"\nOverall Aesthetic Score: {overall_score:.4f}")

# Visualize attention maps for each attribute
for i, attribute_name in enumerate(attribute_names):
    attention_map = attention_maps[i]  # Get the attention map for this attribute
    score = attribute_scores[0, i].item()
    weight = weights[0, i].item()

    save_path = f'attention_visualizations/{attribute_name}.png'  # Save path for the attention visualization
    visualize_attention(attention_map, original_image, attribute_name, score, weight, save_path)
