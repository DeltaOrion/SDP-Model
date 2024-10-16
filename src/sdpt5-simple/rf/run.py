import os
import random

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm

# Configuration
RANDOM_SEED = 42
BATCH_SIZE = 1024
MAX_TOKEN_LENGTH = 1024
N_ESTIMATORS = 100  # Number of trees in the forest
MAX_DEPTH = None  # Max depth of the trees
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set random seeds for reproducibility
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# Step 1: Load Data

input_directory = '../../../dataset'
nodes_df = pd.read_csv(os.path.join(input_directory, 'nodes.csv'))

# Step 2: Initialize CodeT5+ Model and Load Precomputed Embeddings
embedding_file = f"{input_directory}-{MAX_TOKEN_LENGTH}.pt"  # Save the embeddings with .pt extension

if os.path.exists(embedding_file):
    # Load precomputed embeddings from the file
    print(f"Loading precomputed embeddings from {embedding_file}...")
    node_embeddings = torch.load(embedding_file)
else:
    raise FileNotFoundError(f"Embeddings file {embedding_file} not found. Please generate embeddings first.")

# Step 3: Prepare the dataset
print("Preparing the dataset...")
embeddings = []
labels = []

for idx, row in tqdm(nodes_df.iterrows(), total=len(nodes_df)):
    node_id = row['ID']
    embedding = node_embeddings[node_id]
    embeddings.append(embedding)
    label = int(row['NUMBER-OF-BUGS'] >= 1)  # Binary classification: 1 if buggy, 0 otherwise
    labels.append(label)

embeddings = np.stack([embedding.numpy() for embedding in embeddings])  # Convert to numpy array
labels = np.array(labels)

# Step 4: Split Data into Training, Validation, and Test Sets
num_samples = len(embeddings)
indices = np.arange(num_samples)
np.random.shuffle(indices)

train_split = int(0.8 * num_samples)
val_split = int(0.9 * num_samples)

train_indices = indices[:train_split]
val_indices = indices[train_split:val_split]
test_indices = indices[val_split:]

train_embeddings = embeddings[train_indices]
train_labels = labels[train_indices]

val_embeddings = embeddings[val_indices]
val_labels = labels[val_indices]

test_embeddings = embeddings[test_indices]
test_labels = labels[test_indices]

print(f'Total samples: {num_samples}')
print(f'Training samples: {len(train_embeddings)}')
print(f'Validation samples: {len(val_embeddings)}')
print(f'Test samples: {len(test_embeddings)}')

# Step 5: Train the Random Forest Classifier
print("Training the Random Forest Classifier...")
rf_classifier = RandomForestClassifier(random_state=RANDOM_SEED, class_weight='balanced')

rf_classifier.fit(train_embeddings, train_labels)
# Step 6: Save the trained Random Forest model
model_file = 'random_forest_model.pkl'
joblib.dump(rf_classifier, model_file)
print(f"Random Forest model saved as {model_file}")
