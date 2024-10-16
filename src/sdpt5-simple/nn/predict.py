import os
import random
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

from model import SimpleClassifier

# Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EMBEDDING_DIM = 256  # Adjust if using a different model
BATCH_SIZE = 32  # Adjust depending on your system's capacity
MAX_TOKEN_LENGTH = 512  # Same as training
RANDOM_SEED = 42

# Set random seeds for reproducibility
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# Load the unseen data
input_directory = '../../../unseen'
unseen_data_df = pd.read_csv(os.path.join(input_directory, 'nodes.csv'))  # Assuming the unseen data is in a CSV

# Initialize CodeT5+ Model and Tokenizer for embedding generation
checkpoint = "Salesforce/codet5p-110m-embedding"
tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
embedding_model = AutoModel.from_pretrained(checkpoint, trust_remote_code=True)
embedding_model.to(DEVICE)
embedding_model.eval()  # Set to evaluation mode since we're only extracting embeddings

# Precompute embeddings for the unseen data
print("Precomputing embeddings for unseen data...")
unseen_embeddings = []

with torch.no_grad():
    for idx, row in tqdm(unseen_data_df.iterrows(), total=len(unseen_data_df)):
        code_file = row['METHOD-SOURCE-FILE']
        code_file_path = os.path.join(input_directory, code_file)
        with open(code_file_path, 'r') as f:
            code = f.read()
        tokens = tokenizer(code, return_tensors='pt', truncation=True, max_length=MAX_TOKEN_LENGTH)
        tokens = {key: val.to(DEVICE) for key, val in tokens.items()}
        outputs = embedding_model(**tokens)
        unseen_embeddings.append(outputs.squeeze(0).cpu())  # Store embeddings in CPU memory

unseen_embeddings = torch.stack(unseen_embeddings)

# Prepare DataLoader
from torch.utils.data import Dataset, DataLoader

class UnseenDataset(Dataset):
    def __init__(self, embeddings):
        self.embeddings = embeddings

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx]

unseen_dataset = UnseenDataset(unseen_embeddings)
unseen_loader = DataLoader(unseen_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Load the trained model
model = SimpleClassifier(EMBEDDING_DIM)
model.to(DEVICE)

# Load the best model checkpoint
model.load_state_dict(torch.load('best_model.pth', map_location=DEVICE))
model.eval()

# Make predictions on the unseen data
predictions = []

with torch.no_grad():
    for batch_embeddings in tqdm(unseen_loader, desc='Predicting'):
        batch_embeddings = batch_embeddings.to(DEVICE)
        out = model(batch_embeddings).squeeze(-1)  # Get model outputs (logits)
        probs = torch.sigmoid(out).cpu().numpy()  # Convert logits to probabilities
        predictions.extend(probs)

# Save predictions along with method signatures
output_df = unseen_data_df[['SIGNATURE']].copy()
output_df['PREDICTION'] = predictions  # Add the predicted probabilities
output_df.to_csv('predictions.csv', index=False)

print("Predictions saved to predictions.csv")