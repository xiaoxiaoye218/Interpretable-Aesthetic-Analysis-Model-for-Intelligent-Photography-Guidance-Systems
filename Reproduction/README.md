### README for Aesthetic Evaluation Model

---

### Overview

This project aims to replicate the **Aesthetic Evaluation Model** based on the paper *"Interpretable Aesthetic Analysis Model for Intelligent Photography"*. The model focuses on decomposing the overall aesthetic score of an image as a combination of individual attribute scores. The architecture consists of a **Hyper-network** that learns the weights for each attribute score, along with an **attention mechanism** that enables interpretability by highlighting important regions of the image for each attribute.

The model can be broken down into three key modules:
1. **CNN for Feature Extraction**: Extracts high-level feature representations from the input image.
2. **Attribute Score Estimators**: Predicts scores for each aesthetic attribute based on attribute-specific features.
3. **Aesthetic Decomposition Network (Hyper-network)**: Learns the contribution (weights) of each attribute to the overall aesthetic score.

---

### Directory Structure

```text
project_root/
│
├── data/                                  # Data directory
│   ├── train.csv                          # Training data
│   ├── val.csv                            # Validation data
│   ├── test.csv                           # Test data
│   └── images/                            # Image folder for input data
│
├── models/                                # Model definitions
│   ├── cnn_feature_extractor.py           # CNN for feature extraction
│   ├── attribute_score_predictor.py       # Attribute score estimation
│   ├── aesthetic_decomposition_network.py # Hyper-network for weight learning
│   └── attention_module.py                # Spatial attention module
│
├── utils/                                 # Utility functions
│   ├── data_loader.py                     # Data loading and preprocessing
│   ├── metrics.py                         # Evaluation metrics like MSE, correlation
│   └── visualization.py                   # Grad-CAM and attention visualization
│
├── train.py                               # Main training script
├── evaluate.py                            # Evaluation script for model performance
├── README.md                              # Project description and instructions (this file)
├── config.json                            # Configuration file for hyperparameters
└── requirements.txt                       # Python package dependencies
```

---

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/aesthetic-evaluation.git
   cd aesthetic-evaluation
   ```

2. Create a virtual environment and install dependencies:

   ```bash
   conda create --name aesthetic-eval python=3.8
   conda activate aesthetic-eval
   pip install -r requirements.txt
   ```

---

### Configuration

The training process is controlled via the `config.json` file:

```json
{
  "train_path": "data/train.csv",
  "val_path": "data/val.csv",
  "img_folder_path": "data/images",
  "batch_size": 32,
  "n_epochs": 50,
  "use_cuda": true,
  "learning_rate": 1e-4,
  "lambda_att": 0.5,
  "lambda_mi": 0.1,
  "save_path": "checkpoints/"
}
```

Make sure to adjust the file paths and hyperparameters in the `config.json` file based on your environment and dataset.

---

### Training

To train the model, use the following command:

```bash
python train.py --config_file_path config.json
```

This script will train the full model (including the attention module, attribute score estimators, and the hyper-network) on the training set and evaluate it on the validation set at each epoch. The model weights will be saved in the `checkpoints/` directory.

---

### Evaluation

To evaluate the model on the test dataset:

```bash
python evaluate.py --config_file_path config.json
```

The evaluation script will output the Mean Squared Error (MSE) and correlation metrics for both the overall score and individual attributes.

---

### Model Architecture

1. **CNN Feature Extraction**:
   - Uses a pre-trained ResNet backbone to extract high-level feature maps from input images.
   
2. **Attention Module**:
   - Applies spatial attention to the feature maps to focus on regions important for each attribute.
   - Outputs an attention-weighted feature map.

3. **Attribute Score Estimators**:
   - Each attribute has its own score estimator (a fully connected layer).
   - Predicts scores for attributes such as `ColorHarmony`, `Content`, `DoF`, etc.

4. **Aesthetic Decomposition Network (Hyper-network)**:
   - Learns the weight of each attribute in the overall aesthetic score using a small fully-connected network (Hyper-network).
   - The weights are dynamically generated for each image and determine how much each attribute contributes to the overall score.

---

### Key Files

1. **`cnn_feature_extractor.py`**: Defines the CNN-based feature extractor (e.g., ResNet).
2. **`attribute_score_predictor.py`**: Implements the attribute score prediction layers.
3. **`aesthetic_decomposition_network.py`**: Defines the hyper-network that learns the attribute weights.
4. **`attention_module.py`**: Implements the spatial attention mechanism.
5. **`train.py`**: Main script to train the model.
6. **`evaluate.py`**: Script for evaluating the model on test data.
7. **`config.json`**: Hyperparameter configuration file.

---

### Visualization

To visualize the attention maps and model predictions using Grad-CAM, use the `visualization.py` utility script. This script highlights the regions of the image that most contribute to the predicted attribute scores.

```bash
python visualize_attention.py --image_path data/images/sample.jpg --checkpoint checkpoints/model.pth
```

---

### Notes

- **Grad-CAM**: The attention maps can be visualized using Grad-CAM to provide interpretability.
- **Mutual Information Loss**: The mutual information regularizer improves the accuracy of the attention maps by maximizing the mutual information between the attention maps and attribute scores.
- **End-to-End Training**: The entire framework is trained end-to-end, and gradients flow through the hyper-network into the attribute score estimators and attention modules.

---

### Contact

For any questions or issues, feel free to contact me at [your-email@example.com].

