import numpy as np
import joblib
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, accuracy_score, roc_curve
import matplotlib.pyplot as plt

# Load the model, scaler, and test set
model = joblib.load('random_forest_model.joblib')
scaler = joblib.load('scaler.joblib')
test_data = joblib.load('test_data.joblib')  # Load the test data saved from the training script

# Extract X_test and y_test from the loaded test data
X_test = test_data['X_test']
y_test = test_data['y_test']

# Get model predictions and probabilities
probs = model.predict_proba(X_test)[:, 1]  # Probability of the positive class (buggy)
all_labels = y_test

# Compute and print evaluation metrics for different thresholds
thresholds = np.arange(0.1, 1.1, 0.1)
metrics = []

print("\nEvaluation Metrics for Different Thresholds")
print("Threshold | Accuracy | Precision | Recall | F1 Score")

for threshold in thresholds:
    preds = (probs >= threshold).astype(int)
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
    roc_auc = roc_auc_score(all_labels, probs)
    print(f"\nROC-AUC: {roc_auc:.4f}")
except ValueError:
    roc_auc = None
    print("\nROC-AUC could not be computed (only one class present in y_true).")

probs = model.predict_proba(X_test)[:, 1]
all_labels = y_test

# Save the predicted probabilities and true labels
np.save('sdp_hand_probs.npy', probs)
np.save('sdp_hand_labels.npy', all_labels)

# Plot ROC Curve
fpr, tpr, _ = roc_curve(all_labels, probs)
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