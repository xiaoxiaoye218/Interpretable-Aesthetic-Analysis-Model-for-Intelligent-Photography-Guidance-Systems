import torch
from utils.data_loader import create_dataloader
from models.cnn_feature_extractor import CNNFeatureExtractor
from models.attribute_score_predictor import AttributeScorePredictorWithAttention
from models.aesthetic_decomposition_network import AestheticDecompositionNetwork
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
attribute_predictor_with_attention = AttributeScorePredictorWithAttention(num_attributes=11).to(device)
decomposition_network = AestheticDecompositionNetwork(num_attributes=11).to(device)

# Load the trained model (specify the epoch you want to use)
model_path = os.path.join(config['save_path'], f"epoch_{config['eval_epoch']}.pth")
checkpoint = torch.load(model_path, map_location=device)

cnn.load_state_dict(checkpoint['cnn_state_dict'])
attribute_predictor_with_attention.load_state_dict(checkpoint['attribute_predictor_state_dict'])
decomposition_network.load_state_dict(checkpoint['decomposition_network_state_dict'])

cnn.eval()
attribute_predictor_with_attention.eval()
decomposition_network.eval()

# Load the test data
test_loader = create_dataloader(config['test_path'], config['img_folder_path'], batch_size=config['batch_size'], is_train=False)

# Evaluation loop
for images, labels in test_loader:
    images, labels = images.to(device), labels.to(device)

    with torch.no_grad():
        features = cnn(images)
        attribute_scores, attention_maps = attribute_predictor_with_attention(features)
        weights = decomposition_network(features)
        overall_score = torch.sum(attribute_scores * weights, dim=1)

    # Print the evaluation result for each batch
    print(f"Overall Aesthetic Score: {overall_score.item():.4f}")
