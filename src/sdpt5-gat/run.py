import os
import random
from collections import Counter

import numpy as np
import pandas as pd
import torch
from torch.nn import BCEWithLogitsLoss
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

from graph_dataset import GraphDataset
from model import GAT

# Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EMBEDDING_DIM = 256  # Adjust if using a different model
BATCH_SIZE = 1  # Set to 1 since graphs may have variable sizes
EPOCHS = 100
LEARNING_RATE = 0.0001
PATIENCE = 20  # For early stopping
RANDOM_SEED = 42
MAX_TOKEN_LENGTH = 512

if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Set random seeds for reproducibility
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# Step 1: Load Data
print("Loading data....")
input_directory = 'dataset'
nodes_df = pd.read_csv(os.path.join(input_directory, 'nodes.csv'))
edges_df = pd.read_csv(os.path.join(input_directory, 'edges.csv'))

# Step 2: Initialize CodeT5+ Model and Tokenizer
checkpoint = "Salesforce/codet5p-110m-embedding"
tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
embedding_model = AutoModel.from_pretrained(checkpoint, trust_remote_code=True)
embedding_model.to(DEVICE)
embedding_model.train()

embedding_file = f"{input_directory}-{MAX_TOKEN_LENGTH}.pt"  # Save the embeddings with .pt extension

# Check if the embeddings file exists
if os.path.exists(embedding_file):
    # Load precomputed embeddings from the file
    print(f"Loading precomputed embeddings from {embedding_file}...")
    node_embeddings = torch.load(embedding_file)
else:
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
# Step 2: Build Graphs using Precomputed Embeddings
print("Building Graphs with Precomputed Embeddings...")
graph_builder = GraphDataset(input_directory='dataset', node_embeddings=node_embeddings, device=DEVICE)
graphs = graph_builder.build_graphs()

# Step 5: Split Data into Training, Validation, and Test Sets
random.shuffle(graphs)
num_graphs = len(graphs)
train_split = int(0.8 * num_graphs)
val_split = int(0.9 * num_graphs)

train_graphs = graphs[:train_split]
val_graphs = graphs[train_split:val_split]
test_graphs = graphs[val_split:]

print(f'Total graphs: {num_graphs}')
print(f'Training graphs: {len(train_graphs)}')
print(f'Validation graphs: {len(val_graphs)}')
print(f'Test graphs: {len(test_graphs)}')

# Prepare DataLoaders
train_loader = DataLoader(train_graphs, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_graphs, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_graphs, batch_size=BATCH_SIZE, shuffle=False)

# Step 5: Compute Class Weights
train_labels = []
for graph in train_graphs:
    train_labels.extend(graph.y.cpu().numpy().tolist())

class_counts = Counter(train_labels)
total_count = sum(class_counts.values())
class_weights = {cls: total_count / count for cls, count in class_counts.items()}

# Convert to tensor
weights = torch.tensor([class_weights.get(0, 1.0), class_weights.get(1, 1.0)], dtype=torch.float).to(DEVICE)
print(f'Class Counts: {class_counts}')
print(f'Class Weights: {class_weights}')

# Define criterion with class weights
criterion = BCEWithLogitsLoss(pos_weight=weights[1])

model = GAT(EMBEDDING_DIM, 1)  # Binary classification
model.to(DEVICE)

# Combine parameters of both models
optimizer = torch.optim.Adam([
    {'params': model.parameters(), 'lr': LEARNING_RATE},
], weight_decay=5e-4)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

# Early Stopping Parameters
best_val_loss = float('inf')
patience_counter = 0

# Step 5: Training Loop
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{EPOCHS} - Training'):
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index).squeeze(-1)
        loss = criterion(out, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)
    scheduler.step()

    # Validation Loop
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f'Epoch {epoch + 1}/{EPOCHS} - Validation'):
            out = model(batch.x, batch.edge_index).squeeze(-1)
            loss = criterion(out, batch.y.float())
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    print(f'Epoch {epoch + 1}/{EPOCHS}, Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}')

    # Early Stopping Check
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        # Save the best model
        torch.save({
            'gat_state_dict': model.state_dict(),
        }, 'best_model.pth')
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print("Early stopping triggered.")
            break
