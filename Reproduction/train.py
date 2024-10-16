import torch
import torch.optim as optim
import torch.nn as nn
from utils.data_loader import create_dataloader
from models.cnn_feature_extractor import CNNFeatureExtractor
from models.attribute_score_predictor import AttributeScorePredictorWithAttention
from models.aesthetic_decomposition_network import AestheticDecompositionNetwork
from utils.metrics import mean_squared_error
from tqdm import tqdm
import os
import json

# Load configuration
config_path = 'config.json'
if not os.path.exists(config_path):
    raise FileNotFoundError(f"Configuration file not found at {config_path}")

config = json.load(open(config_path))

# Load data
train_loader = create_dataloader(config['train_path'], config['img_folder_path'], batch_size=config['batch_size'])
val_loader = create_dataloader(config['val_path'], config['img_folder_path'], batch_size=config['batch_size'], is_train=False)

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() and config['use_cuda'] else 'cpu')

# Initialize models and move them to the appropriate device
cnn = CNNFeatureExtractor().to(device)
attribute_predictor_with_attention = AttributeScorePredictorWithAttention(num_attributes=11).to(device)
decomposition_network = AestheticDecompositionNetwork(num_attributes=11).to(device)

# Loss function
criterion = nn.MSELoss()

# Optimizer
optimizer = optim.Adam(
    list(cnn.parameters()) + list(attribute_predictor_with_attention.parameters()) +
    list(decomposition_network.parameters()),
    lr=config['learning_rate']
)

# Training loop
for epoch in tqdm(range(config['n_epochs']), desc="Epochs Progress"):
    cnn.train()
    total_loss = 0

    for batch_idx, (images, labels) in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch+1}")):
        # Move images and labels to device
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        features = cnn(images)
        attribute_scores, attention_maps = attribute_predictor_with_attention(features)
        weights = decomposition_network(features)
        overall_score = torch.sum(attribute_scores * weights, dim=1)

        # Loss computation
        overall_loss = criterion(overall_score, labels[:, -1])  # Overall score loss
        attribute_loss = criterion(attribute_scores, labels[:, :-1])  # Attribute score loss

        # Combine the losses
        total_loss_batch = overall_loss + config['lambda_att'] * attribute_loss
        optimizer.zero_grad()
        total_loss_batch.backward()
        optimizer.step()
        total_loss += total_loss_batch.item()

        if batch_idx % 10 == 0:
            print(f'Batch {batch_idx}, Loss: {total_loss_batch.item():.4f}')

    avg_loss = total_loss / len(train_loader)
    print(f'Epoch {epoch + 1}, Average Loss: {avg_loss:.4f}')

    # Validation
    cnn.eval()
    with torch.no_grad():
        val_loss = 0
        for images, labels in tqdm(val_loader, desc=f"Validation Epoch {epoch+1}"):
            images, labels = images.to(device), labels.to(device)
            features = cnn(images)
            attribute_scores, attention_maps = attribute_predictor_with_attention(features)
            weights = decomposition_network(features)
            overall_score = torch.sum(attribute_scores * weights, dim=1)

            # Validation loss
            overall_loss = criterion(overall_score, labels[:, -1])
            val_loss += overall_loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f'Validation Loss for Epoch {epoch + 1}: {avg_val_loss:.4f}')

    # Save model after each epoch
    os.makedirs(config['save_path'], exist_ok=True)
    torch.save({
        'epoch': epoch + 1,
        'cnn_state_dict': cnn.state_dict(),
        'attribute_predictor_state_dict': attribute_predictor_with_attention.state_dict(),
        'decomposition_network_state_dict': decomposition_network.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
    }, f"{config['save_path']}/epoch_{epoch + 1}.pth")
    print(f"Model saved to {config['save_path']}/epoch_{epoch + 1}.pth")
