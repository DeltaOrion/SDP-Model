import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
import joblib
import numpy as np

# Load dataset
data = pd.read_csv('../../../hand-data/method-p.csv')

# Convert 'Number of Bugs' to binary classification (>= 1 becomes 1, else 0)
data['Number of Bugs'] = (data['Number of Bugs'] >= 1).astype(int)

# Map project names (strings) to integers
project_mapping = {project: idx for idx, project in enumerate(data['Project'].unique())}
data['Project'] = data['Project'].map(project_mapping)

# Initialize lists for training and validation/test data
train_list = []
val_test_list = []

# Split the data by 'Project'
for project, project_data in data.groupby('Project'):
    # Split project data (80% train, 20% test/val)
    X = project_data.drop(columns=['Hash', 'LongName', 'Parent', 'Number of Bugs'])  # Keep 'Project' for now
    y = project_data['Number of Bugs']

    X_train, X_val_test, y_train, y_val_test = train_test_split(X, y, test_size=0.1, random_state=42)

    train_list.append(pd.concat([X_train, y_train], axis=1))
    val_test_list.append(pd.concat([X_val_test, y_val_test], axis=1))

# Combine project splits into final training and validation/test sets
train_data = pd.concat(train_list)
val_test_data = pd.concat(val_test_list)

# Separate features and labels (keep 'Project' for test/validation set)
X_train = train_data.drop(columns=['Number of Bugs'])
y_train = train_data['Number of Bugs']
X_val_test = val_test_data.drop(columns=['Number of Bugs'])  # Keep 'Project' for later but exclude from scaling
y_val_test = val_test_data['Number of Bugs']

# Exclude the 'Project' column from scaling
columns_to_scale = X_train.columns.difference(['Project'])

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train[columns_to_scale])
X_val_test_scaled = scaler.transform(X_val_test[columns_to_scale])

# Combine scaled features with 'Project' for model input (if needed)
X_train_final = np.column_stack([X_train_scaled, X_train['Project']])
X_val_test_final = np.column_stack([X_val_test_scaled, X_val_test['Project']])

# Initialize RandomForest model
model = RandomForestClassifier(random_state=42, class_weight='balanced')

# Fit model to the training data
model.fit(X_train_final, y_train)

# Save the trained model and scaler
joblib.dump(model, 'random_forest_model_project_split.joblib')
joblib.dump(scaler, 'scaler_project_split.joblib')

# Save the validation/test set for evaluation and the project mappings
val_test_data = {"X_val_test": X_val_test_final, "y_val_test": y_val_test, "Projects": val_test_data['Project'].values}
joblib.dump(val_test_data, 'val_test_data_project_split.joblib')
joblib.dump(project_mapping, 'project_mapping.joblib')  # Save the project name to int mapping

print("Model, scaler, validation/test data, and project mappings saved.")

# =========================== EVALUATION PART ===========================

# Reverse the project mapping (int to original project names)
reverse_project_mapping = {v: k for k, v in project_mapping.items()}

# Extract X_val_test, y_val_test, and project labels
X_val_test = val_test_data['X_val_test']
y_val_test = val_test_data['y_val_test']
projects = val_test_data['Projects']

# Get model predictions and probabilities
probs = model.predict_proba(X_val_test)[:, 1]  # Probability of the positive class (buggy)
all_labels = y_val_test

# Iterate through each project
project_results = {}

for project in np.unique(projects):
    project_indices = (projects == project)
    project_labels = all_labels[project_indices]
    project_probs = probs[project_indices]

    best_f1 = 0
    best_threshold = 0
    best_metrics = {}

    # Iterate over thresholds to find the best F1 score
    for threshold in np.arange(0.0, 1.01, 0.01):
        preds = (project_probs >= threshold).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(project_labels, preds, average='binary', pos_label=1)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            best_metrics = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'threshold': threshold
            }

    # Use the original project name for output
    project_name = reverse_project_mapping[project]
    project_results[project_name] = best_metrics
    print(f"Project: {project_name}, Best F1: {best_f1:.4f} at threshold {best_threshold:.2f}")

# Compute and print overall ROC-AUC across all projects
roc_auc = roc_auc_score(all_labels, probs)
print(f"\nOverall ROC-AUC: {roc_auc:.4f}")

# Save the results
results_df = pd.DataFrame.from_dict(project_results, orient='index')
results_df.to_csv('project_split_evaluation_results.csv', index_label='Project')

print("Best F1 scores and thresholds for each project saved to 'project_split_evaluation_results.csv'.")
