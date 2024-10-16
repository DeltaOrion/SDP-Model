import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, accuracy_score, roc_curve
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from graph_dataset import GraphDataset
from model import GAT

# Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EMBEDDING_DIM = 256  # Adjust if using a different model
BATCH_SIZE = 1  # Since graphs may have variable sizes
MAX_NUMBER_OF_TOKENS = 512

# Load precomputed embeddings and data
input_directory = 'dataset'
nodes_df = pd.read_csv(os.path.join(input_directory, 'nodes.csv'))
edges_df = pd.read_csv(os.path.join(input_directory, 'edges.csv'))

embedding_file = f"{input_directory}-{MAX_NUMBER_OF_TOKENS}.pt"
if os.path.exists(embedding_file):
    print(f"Loading precomputed embeddings from {embedding_file}...")
    node_embeddings = torch.load(embedding_file)
else:
    raise FileNotFoundError(f"Precomputed embeddings file {embedding_file} not found.")

# Build Graphs using Precomputed Embeddings
print("Building Graphs with Precomputed Embeddings...")
# Create an instance of GraphBuilder
graph_builder = GraphDataset(input_directory='dataset', node_embeddings=node_embeddings, device=DEVICE)
graphs = graph_builder.build_graphs()

num_graphs = len(graphs)
val_split = int(0.9 * num_graphs)
test_graphs = graphs[val_split:]

print(f'Total graphs: {num_graphs}')
print(f'Test graphs: {len(test_graphs)}')

test_loader = DataLoader(test_graphs, batch_size=BATCH_SIZE, shuffle=False)

model = GAT(EMBEDDING_DIM, 1)  # Binary classification
model.to(DEVICE)

checkpoint = torch.load('best_model.pth', map_location=DEVICE)
model.load_state_dict(checkpoint['gat_state_dict'])
model.eval()

all_preds = []
all_labels = []
all_probs = []

with torch.no_grad():
    for batch in tqdm(test_loader, desc='Testing'):
        out = model(batch.x, batch.edge_index).squeeze(-1)  # Get model predictions
        probs = torch.sigmoid(out).cpu().numpy()  # Convert logits to probabilities for class 1
        labels = batch.y.cpu().numpy()
        all_probs.extend(probs)
        all_labels.extend(labels)

all_labels = np.array(all_labels)
all_probs = np.array(all_probs)

# Compute and print evaluation metrics for different thresholds
thresholds = np.arange(0.1, 1.1, 0.1)  # Thresholds from 0.0 to 1.0 with step of 0.1
metrics = []

print("\nEvaluation Metrics for Different Thresholds")
print("Threshold | Accuracy | Precision | Recall | F1 Score")

for threshold in thresholds:
    preds = (all_probs >= threshold).astype(int)
    accuracy = accuracy_score(all_labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, preds, average='binary', pos_label=1)

    metrics.append({
        'threshold': threshold,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    })

    print(f'{threshold:.2f}      | {accuracy:.4f}  | {precision:.4f}  | {recall:.4f}  | {f1:.4f}')

# Compute ROC-AUC
try:
    roc_auc = roc_auc_score(all_labels, all_probs)
    print(f"\nROC-AUC: {roc_auc:.4f}")
except ValueError:
    roc_auc = None
    print("\nROC-AUC could not be computed (only one class present in y_true).")


np.save('sdpt5_gat_probs.npy', all_probs)
np.save('sdpt5_gat_labels.npy', all_labels)

# Plot ROC Curve
fpr, tpr, _ = roc_curve(all_labels, all_probs)
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
