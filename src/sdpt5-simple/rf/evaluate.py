import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from joblib import load
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, accuracy_score, roc_curve
from tqdm import tqdm

# Configuration
RANDOM_SEED = 42
BATCH_SIZE = 32  # Batch size doesn't apply to scikit-learn models, but we can still use it for DataLoader
EMBEDDING_DIM = 256  # This should match your embedding dimension
MAX_TOKEN_LENGTH = 1024

# Set random seeds for reproducibility
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# Load precomputed embeddings and data
input_directory = '../../../dataset'
nodes_df = pd.read_csv(os.path.join(input_directory, 'nodes.csv'))

embedding_file = f"{input_directory}-{MAX_TOKEN_LENGTH}.pt"  # Save the embeddings with .pt extension
if os.path.exists(embedding_file):
    print(f"Loading precomputed embeddings from {embedding_file}...")
    node_embeddings = torch.load(embedding_file)
else:
    raise FileNotFoundError(f"Precomputed embeddings file {embedding_file} not found.")

# Prepare the dataset
print("Preparing the dataset...")
embeddings = []
labels = []

for idx, row in tqdm(nodes_df.iterrows(), total=len(nodes_df)):
    node_id = row['ID']
    embedding = node_embeddings[node_id]
    embeddings.append(embedding.numpy())  # Convert to numpy array for sklearn
    label = int(row['NUMBER-OF-BUGS'] >= 1)
    labels.append(label)

embeddings = np.stack(embeddings)  # Convert list of arrays to numpy matrix
labels = np.array(labels)

# Split Data into Training, Validation, and Test Sets
num_samples = len(embeddings)
indices = np.arange(num_samples)
np.random.shuffle(indices)

train_split = int(0.8 * num_samples)
val_split = int(0.9 * num_samples)

test_indices = indices[val_split:]

test_embeddings = embeddings[test_indices]
test_labels = labels[test_indices]

print(f'Total samples: {num_samples}')
print(f'Test samples: {len(test_embeddings)}')

# Load the trained RandomForestClassifier model
rf_model_path = 'random_forest_model.pkl'
print(f"Loading trained Random Forest model from {rf_model_path}...")
rf_classifier = load(rf_model_path)  # Load the model using joblib

# Perform predictions
print("Performing predictions on the test set...")
test_probs = rf_classifier.predict_proba(test_embeddings)[:, 1]  # Get probabilities for the positive class
test_preds = rf_classifier.predict(test_embeddings)  # Get class predictions

# Evaluation
print("\nEvaluation Metrics for Different Thresholds")
print("Threshold | Accuracy | Precision | Recall | F1 Score")

thresholds = np.arange(0.1, 1.1, 0.1)  # Thresholds for binary classification
for threshold in thresholds:
    binary_preds = (test_probs >= threshold).astype(int)
    accuracy = accuracy_score(test_labels, binary_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(test_labels, binary_preds, average='binary', zero_division=0)

    print(f'{threshold:.2f}      | {accuracy:.4f}  | {precision:.4f}  | {recall:.4f}  | {f1:.4f}')

# Compute ROC-AUC
roc_auc = roc_auc_score(test_labels, test_probs)
print(f"\nROC-AUC: {roc_auc:.4f}")

np.save('sdpt5_simple_rf_probs.npy', test_probs)
np.save('sdpt5_simple_rf_labels.npy', test_labels)

# Plot ROC Curve
fpr, tpr, _ = roc_curve(test_labels, test_probs)
plt.figure()
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()