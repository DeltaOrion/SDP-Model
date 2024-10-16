import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from joblib import load

# Configuration
MAX_TOKEN_LENGTH = 512  # Must match what was used for training
RANDOM_SEED = 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set random seed for reproducibility
np.random.seed(RANDOM_SEED)

# Load the unseen data
input_directory = '../../../unseen'
unseen_data_df = pd.read_csv(os.path.join(input_directory, 'nodes.csv'))  # Assuming unseen data is in CSV format

# Load precomputed embeddings for unseen data
embedding_file = f"{input_directory}-{MAX_TOKEN_LENGTH}.pt"  # Must match the format used for training embeddings
if os.path.exists(embedding_file):
    print(f"Loading precomputed embeddings from {embedding_file}...")
    node_embeddings = torch.load(embedding_file)
else:
    raise FileNotFoundError(f"Precomputed embeddings file {embedding_file} not found.")

# Prepare the dataset for prediction
print("Preparing the dataset for prediction...")
unseen_embeddings = []

for idx, row in tqdm(unseen_data_df.iterrows(), total=len(unseen_data_df)):
    node_id = row['ID']
    embedding = node_embeddings[node_id]
    unseen_embeddings.append(embedding.numpy())  # Convert to numpy array

unseen_embeddings = np.stack(unseen_embeddings)  # Convert to numpy matrix

# Load the trained Random Forest model
rf_model_path = 'best_model.pkl'
print(f"Loading trained Random Forest model from {rf_model_path}...")
rf_classifier = load(rf_model_path)  # Load the model using joblib

# Perform predictions on the unseen data
print("Performing predictions on the unseen data...")
unseen_probs = rf_classifier.predict_proba(unseen_embeddings)[:, 1]  # Get probabilities for the positive class
unseen_preds = rf_classifier.predict(unseen_embeddings)  # Get class predictions

# Save predictions along with the method signatures
output_df = unseen_data_df[['SIGNATURE']].copy()  # Assuming unseen data has a 'METHOD-SIGNATURE' column
output_df['PREDICTION_PROB'] = unseen_probs  # Add predicted probabilities
output_df['PREDICTION_CLASS'] = unseen_preds  # Add predicted class labels (1 for buggy, 0 for non-buggy)

# Save the predictions to a CSV file
output_file = 'predictions.csv'
output_df.to_csv(output_file, index=False)
print(f"Predictions saved to {output_file}")