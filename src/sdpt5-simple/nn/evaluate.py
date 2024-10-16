import os
import random
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, accuracy_score, roc_curve
from tqdm import tqdm
import matplotlib.pyplot as plt

from model import SimpleClassifier

# Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EMBEDDING_DIM = 256  # Adjust if using a different model
BATCH_SIZE = 32  # Same as in training script
RANDOM_SEED = 42
MAX_TOKEN_LENGTH = 512

# Set random seeds for reproducibility
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# Load precomputed embeddings and data
input_directory = '../../../dataset'
nodes_df = pd.read_csv(os.path.join(input_directory, 'nodes.csv'))

embedding_file = f"{input_directory}-{MAX_TOKEN_LENGTH}.pt"
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
    embeddings.append(embedding)
    label = int(row['NUMBER-OF-BUGS'] >= 1)
    labels.append(label)

embeddings = torch.stack(embeddings)
labels = torch.tensor(labels, dtype=torch.float32)

# Split Data into Training, Validation, and Test Sets
num_samples = len(embeddings)
indices = list(range(num_samples))
random.shuffle(indices)

train_split = int(0.8 * num_samples)
val_split = int(0.9 * num_samples)

train_indices = indices[:train_split]
val_indices = indices[train_split:val_split]
test_indices = indices[val_split:]

test_embeddings = embeddings[test_indices]
test_labels = labels[test_indices]

print(f'Total samples: {num_samples}')
print(f'Test samples: {len(test_embeddings)}')

# Prepare DataLoader
from torch.utils.data import Dataset, DataLoader


class NodeDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


test_dataset = NodeDataset(test_embeddings, test_labels)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = SimpleClassifier(EMBEDDING_DIM)
model.to(DEVICE)

# Load the best model
model.load_state_dict(torch.load('best_model.pth', map_location=DEVICE))
model.eval()

# Evaluation
all_preds = []
all_labels = []
all_probs = []

with torch.no_grad():
    for batch_embeddings, batch_labels in tqdm(test_loader, desc='Testing'):
        batch_embeddings = batch_embeddings.to(DEVICE)
        batch_labels = batch_labels.to(DEVICE)
        out = model(batch_embeddings).squeeze(-1)  # Get model predictions
        probs = torch.sigmoid(out).cpu().numpy()  # Convert logits to probabilities
        labels = batch_labels.cpu().numpy()
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
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, preds, average='binary', pos_label=1,
                                                               zero_division=0)
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

np.save('sdpt5_simple_nn_probs.npy', all_probs)
np.save('sdpt5_simple_nn_labels.npy', all_labels)

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