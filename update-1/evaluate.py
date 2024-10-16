import torch
from torchvision import transforms
from models.cnn_feature_extractor import CNNFeatureExtractor
from models.attribute_score_predictor import AttributeScorePredictorWithAttention
from models.aesthetic_decomposition_network import AestheticDecompositionNetwork
import os
import json
from utils.data_loader import create_dataloader
from utils.metrics import pearson_correlation  # 引入你的 Pearson 相关系数函数

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

# Load data
test_loader = create_dataloader(config['test_path'], config['img_folder_path'], batch_size=config['batch_size'], is_train=False)

# Initialize lists to store groundtruth and predictions
gt_attributes = []
pred_attributes = []
gt_overall_scores = []
pred_overall_scores = []

# Evaluation
with torch.no_grad():
    for batch_idx, (images, labels) in enumerate(test_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        features = cnn(images)
        attribute_scores, attention_maps = attribute_predictor_with_attention(features)
        weights = decomposition_network(features)
        overall_scores = torch.sum(attribute_scores * weights, dim=1)

        # Accumulate groundtruth and predictions
        gt_attributes.append(labels[:, :-1].cpu())
        pred_attributes.append(attribute_scores.cpu())
        gt_overall_scores.append(labels[:, -1].cpu())
        pred_overall_scores.append(overall_scores.cpu())

        # Print results for each image in the batch
        attribute_names = ['BalancingElements', 'ColorHarmony', 'Content', 'DoF', 'Light', 'MotionBlur',
                           'Object', 'Repetition', 'RuleOfThirds', 'Symmetry', 'VividColor']

        for i in range(images.size(0)):  # iterate through the batch
            print(f"\nImage {batch_idx * config['batch_size'] + i + 1}:")
            print("Predicted Attribute Scores:")
            for j, score in enumerate(attribute_scores[i]):
                print(f"{attribute_names[j]}: {score.item():.4f}")

            print("\nPredicted Attribute Weights:")
            for j, weight in enumerate(weights[i]):
                print(f"{attribute_names[j]}: {weight.item():.4f}")

            print(f"\nOverall Aesthetic Score: {overall_scores[i].item():.4f}")

# Concatenate all batches' groundtruth and predictions
gt_attributes = torch.cat(gt_attributes, dim=0)
pred_attributes = torch.cat(pred_attributes, dim=0)
gt_overall_scores = torch.cat(gt_overall_scores, dim=0)
pred_overall_scores = torch.cat(pred_overall_scores, dim=0)

# Compute Pearson correlation for each attribute and overall score
pearson_scores = []
for i in range(pred_attributes.size(1)):  # for each attribute
    pearson_score = pearson_correlation(pred_attributes[:, i], gt_attributes[:, i])
    pearson_scores.append(pearson_score)
    print(f"Pearson correlation for {attribute_names[i]}: {pearson_score:.4f}")

# Pearson correlation for overall score
overall_pearson = pearson_correlation(pred_overall_scores, gt_overall_scores)
pearson_scores.append(overall_pearson)
print(f"\nPearson correlation for Overall Aesthetic Score: {overall_pearson:.4f}")

# Compute and print average Pearson correlation
average_pearson = sum(pearson_scores) / len(pearson_scores)
print(f"\nAverage Pearson correlation across attributes and overall score: {average_pearson:.4f}")
