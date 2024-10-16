import os
import random
from collections import Counter

import numpy as np
import pandas as pd
import torch
from torch.nn import BCEWithLogitsLoss
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import roc_auc_score
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
from model import SimpleClassifier

# Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EMBEDDING_DIM = 256  # Adjust if using a different model
BATCH_SIZE = 1024
EPOCHS = 100
LEARNING_RATE = 0.001
PATIENCE = 5  # For early stopping
RANDOM_SEED = 42
MAX_TOKEN_LENGTH = 512

# Set random seeds for reproducibility
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# Load Data
print("Loading data....")
input_directory = '../../../dataset'
nodes_df = pd.read_csv(os.path.join(input_directory, 'nodes.csv'))

# Initialize CodeT5+ Model and Tokenizer
checkpoint = "Salesforce/codet5p-110m-embedding"
tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
embedding_model = AutoModel.from_pretrained(checkpoint, trust_remote_code=True)
embedding_model.to(DEVICE)
embedding_model.eval()  # Set to evaluation mode since we're only extracting embeddings

embedding_file = f"{input_directory}-{MAX_TOKEN_LENGTH}.pt"  # Save the embeddings with .pt extension

# Check if the embeddings file exists
if os.path.exists(embedding_file):
    # Load precomputed embeddings from the file
    print(f"Loading precomputed embeddings from {embedding_file}...")
    node_embeddings = torch.load(embedding_file)
else:
    # Precompute Tokenizations
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

# Prepare the dataset
print("Preparing the dataset...")
embeddings = []
labels = []
projects = []  # Store the project for each method

for idx, row in tqdm(nodes_df.iterrows(), total=len(nodes_df)):
    node_id = row['ID']
    embedding = node_embeddings[node_id]
    embeddings.append(embedding)
    label = int(row['NUMBER-OF-BUGS'] >= 1)
    labels.append(label)
    projects.append(row['PROJECT'])

embeddings = torch.stack(embeddings)
labels = torch.tensor(labels, dtype=torch.float32)

# Define Dataset Class
class NodeDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]

# n-1 Cross Validation (leave-one-project-out)
project_list = list(set(projects))
results = []

for test_project in project_list:
    print(f"Training and evaluating, leaving out project: {test_project}")

    # Split the data: training data = all projects except the current test project
    train_indices = [i for i, p in enumerate(projects) if p != test_project]
    test_indices = [i for i, p in enumerate(projects) if p == test_project]

    train_embeddings = embeddings[train_indices]
    train_labels = labels[train_indices]
    test_embeddings = embeddings[test_indices]
    test_labels = labels[test_indices]

    # Split training set into train and validation (90% train, 10% validation)
    num_train = len(train_embeddings)
    split_idx = int(num_train * 0.9)

    train_dataset = NodeDataset(train_embeddings[:split_idx], train_labels[:split_idx])
    val_dataset = NodeDataset(train_embeddings[split_idx:], train_labels[split_idx:])
    test_dataset = NodeDataset(test_embeddings, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Compute class weights for training
    train_labels_list = train_labels[:split_idx].cpu().numpy().tolist()
    class_counts = Counter(train_labels_list)
    total_count = sum(class_counts.values())
    class_weights = {cls: total_count / count for cls, count in class_counts.items()}
    weights = torch.tensor([class_weights.get(0.0, 1.0), class_weights.get(1.0, 1.0)], dtype=torch.float32).to(DEVICE)

    # Initialize model and criterion
    model = SimpleClassifier(EMBEDDING_DIM)
    model.to(DEVICE)
    criterion = BCEWithLogitsLoss(pos_weight=weights[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    # Early stopping parameters
    best_val_loss = float('inf')
    patience_counter = 0

    # Training loop
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch_embeddings, batch_labels in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{EPOCHS} - Training'):
            batch_embeddings = batch_embeddings.to(DEVICE)
            batch_labels = batch_labels.to(DEVICE)
            optimizer.zero_grad()
            out = model(batch_embeddings).squeeze(-1)
            loss = criterion(out, batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        scheduler.step()

        # Validation loop
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_embeddings, batch_labels in tqdm(val_loader, desc=f'Epoch {epoch + 1}/{EPOCHS} - Validation'):
                batch_embeddings = batch_embeddings.to(DEVICE)
                batch_labels = batch_labels.to(DEVICE)
                out = model(batch_embeddings).squeeze(-1)
                loss = criterion(out, batch_labels)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f'Epoch {epoch + 1}/{EPOCHS}, Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}')

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save the best model
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("Early stopping triggered.")
                break

    # Evaluation
    model.load_state_dict(torch.load('best_model.pth', map_location=DEVICE))
    model.eval()

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch_embeddings, batch_labels in tqdm(test_loader, desc=f'Testing on {test_project}'):
            batch_embeddings = batch_embeddings.to(DEVICE)
            batch_labels = batch_labels.to(DEVICE)
            out = model(batch_embeddings).squeeze(-1)
            probs = torch.sigmoid(out).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(batch_labels.cpu().numpy())

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Compute AUC
    try:
        roc_auc = roc_auc_score(all_labels, all_probs)
        print(f"\n{test_project} ROC-AUC: {roc_auc:.4f}")
    except ValueError:
        roc_auc = None
        print(f"\nROC-AUC could not be computed for {test_project} (only one class present in y_true).")

    # Save the results
    results.append({'Test Project': test_project, 'AUC': roc_auc})

# Save results to CSV
results_df = pd.DataFrame(results)
results_df.to_csv('n1_cross_project_auc_results.csv', index=False)
print("Results saved to 'n1_cross_project_auc_results.csv'.")