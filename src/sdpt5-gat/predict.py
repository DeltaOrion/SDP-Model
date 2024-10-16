import os

import pandas as pd
import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

from graph_dataset import GraphDataset
from model import GAT

# Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 1
EMBEDDING_DIM = 256  # Adjust if using a different model
MAX_TOKEN_LENGTH = 512

# Load precomputed embeddings
input_directory = 'unseen'
embedding_file = f"{input_directory}.pt"

nodes_df = pd.read_csv(os.path.join(input_directory, 'nodes.csv'))
edges_df = pd.read_csv(os.path.join(input_directory, 'edges.csv'))

# Step 2: Initialize CodeT5+ Model and Tokenizer
checkpoint = "Salesforce/codet5p-110m-embedding"
tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
embedding_model = AutoModel.from_pretrained(checkpoint, trust_remote_code=True)
embedding_model.to(DEVICE)

# Step 3: Precompute Tokenizations
print("Precomputing node embeddings...")
node_embeddings = {}

# Loop through each node and precompute the embeddings
with torch.no_grad():  # No need to fine-tune CodeT5
    for idx, row in tqdm(nodes_df.iterrows(), total=len(nodes_df)):
        code_file = row['METHOD-SOURCE-FILE']
        code_file_path = os.path.join(input_directory, code_file)
        try:
            with open(code_file_path, 'r') as f:
                code = f.read()
            tokens = tokenizer(code, return_tensors='pt', truncation=True, max_length=MAX_TOKEN_LENGTH)
            tokens = {key: val.to(DEVICE) for key, val in tokens.items()}
            outputs = embedding_model(**tokens)
            node_embeddings[row['ID']] = outputs.squeeze(0).cpu()  # Store embeddings in CPU memory
        except FileNotFoundError:
            print(f"File not found: {code_file_path}")
            node_embeddings[row['ID']] = torch.zeros(EMBEDDING_DIM)  # Use zeros for missing files

# Save the precomputed embeddings to a file
print(f"Saving precomputed embeddings to {embedding_file}...")
torch.save(node_embeddings, embedding_file)
# Create an instance of GraphBuilder with is_labeled=False for unseen data
graph_builder = GraphDataset(input_directory=input_directory, node_embeddings=node_embeddings, device=DEVICE,
                             is_labeled=False)

# Build graphs for unseen data
unseen_graphs = graph_builder.build_graphs()
print(f'Total unseen graphs: {len(unseen_graphs)}')

# Create DataLoader for unseen data
unseen_loader = DataLoader(unseen_graphs, batch_size=BATCH_SIZE, shuffle=False)

# Load pre-trained model
model = GAT(EMBEDDING_DIM, 1)  # Binary classification model
model.to(DEVICE)

checkpoint = torch.load('best_model.pth', map_location=DEVICE)
model.load_state_dict(checkpoint['gat_state_dict'])
model.eval()

# Perform prediction on unseen data
all_probs = []
all_signatures = []

with torch.no_grad():
    for batch in tqdm(unseen_loader, desc='Predicting on unseen data'):
        out = model(batch.x, batch.edge_index).squeeze(-1)  # Get model logits
        probs = torch.sigmoid(out).cpu().numpy()  # Convert logits to probabilities for class 1
        all_probs.extend(probs)
        all_signatures.extend(batch.signature)  # Store the signatures

all_signatures = [signature for batch_signatures in all_signatures for signature in batch_signatures]

# Save predictions to a CSV file
output_df = pd.DataFrame({
    'Signature': all_signatures,
    'Predicted_Probability': all_probs
})

output_df.to_csv('predictions.csv', index=False)
print("Predictions saved to 'predictions.csv'")
