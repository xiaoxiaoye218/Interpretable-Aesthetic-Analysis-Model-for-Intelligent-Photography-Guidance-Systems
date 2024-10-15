import torch
from PIL import Image
from torchvision import transforms
from models.cnn_feature_extractor import CNNFeatureExtractor
from models.attribute_score_predictor import AttributeScorePredictor
from models.aesthetic_decomposition_network import AestheticDecompositionNetwork
from models.attention_module import SpatialAttention
import os
import json

# Load configuration
config_path = 'config.json'
if not os.path.exists(config_path):
    raise FileNotFoundError(f"Configuration file not found at {config_path}")

config = json.load(open(config_path))

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() and config['use_cuda'] else 'cpu')

# Initialize models and move them to the appropriate device
cnn = CNNFeatureExtractor().to(device)
attention = SpatialAttention().to(device)
attribute_predictor = AttributeScorePredictor(num_attributes=11).to(device)
decomposition_network = AestheticDecompositionNetwork(num_attributes=11).to(device)

# Load the trained model (specify the epoch you want to use, for example 50)
model_path = os.path.join(config['save_path'], f"epoch_{config['eval_epoch']}.pth")
checkpoint = torch.load(model_path, map_location=device)

cnn.load_state_dict(checkpoint['cnn_state_dict'])
attention.load_state_dict(checkpoint['attention_state_dict'])
attribute_predictor.load_state_dict(checkpoint['attribute_predictor_state_dict'])
decomposition_network.load_state_dict(checkpoint['decomposition_network_state_dict'])

# Set models to evaluation mode
cnn.eval()
attention.eval()
attribute_predictor.eval()
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

# Load and preprocess the image
image_path = 'data/images/farm1_255_19452343093_8ee7e5e375_b.jpg'  # Specify your image path here
image_tensor = load_and_preprocess_image(image_path).to(device)

# Forward pass through the model
with torch.no_grad():
    features = cnn(image_tensor)
    attention_map = attention(features)
    attended_features = features * attention_map
    attribute_scores = attribute_predictor(attended_features)  # Predict attribute scores
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
