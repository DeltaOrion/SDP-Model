import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler

# Load dataset
data = pd.read_csv('../../../hand-data/method-p.csv')

# Convert 'Number of Bugs' to binary classification (>= 1 becomes 1, else 0)
data['Number of Bugs'] = (data['Number of Bugs'] >= 1).astype(int)

# Map project names (strings) to integers
project_mapping = {project: idx for idx, project in enumerate(data['Project'].unique())}
reverse_project_mapping = {v: k for k, v in project_mapping.items()}
data['Project'] = data['Project'].map(project_mapping)

# Initialize results dictionary to store F1 scores for each project pair
results = {project_name: {} for project_name in reverse_project_mapping.values()}


# Function to get the best F1 score for a given threshold range
def get_best_f1(y_true, probs):
    best_f1 = 0
    best_threshold = 0
    for threshold in np.arange(0.0, 1.01, 0.01):
        preds = (probs >= threshold).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, preds, average='binary', pos_label=1)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    return best_f1, best_threshold


# Iterate through each project to train a model on it
for train_project in data['Project'].unique():
    # Separate training data for the current project
    train_data = data[data['Project'] == train_project]
    X_train = train_data.drop(columns=['Project', 'Hash', 'LongName', 'Parent', 'Number of Bugs'])
    y_train = train_data['Number of Bugs']

    # Standardize the training data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Train RandomForest model
    model = RandomForestClassifier(random_state=42, class_weight='balanced')
    model.fit(X_train_scaled, y_train)

    # Evaluate the model on every other project (including itself)
    for test_project in data['Project'].unique():
        test_data = data[data['Project'] == test_project]
        X_test = test_data.drop(columns=['Project', 'Hash', 'LongName', 'Parent', 'Number of Bugs'])
        y_test = test_data['Number of Bugs']

        # Standardize the test data using the scaler fitted on the training project
        X_test_scaled = scaler.transform(X_test)

        # Get model predictions and probabilities
        probs = model.predict_proba(X_test_scaled)[:, 1]  # Probability of the positive class (buggy)

        # Get the best F1 score for the current project pair
        best_f1, best_threshold = get_best_f1(y_test, probs)

        # Record the results using the original project names
        train_project_name = reverse_project_mapping[train_project]
        test_project_name = reverse_project_mapping[test_project]
        results[train_project_name][test_project_name] = best_f1

        print(
            f"{train_project_name} -> {test_project_name} - Best F1: {best_f1:.4f} (at threshold {best_threshold:.2f})")

# Save the results as a CSV file
results_df = pd.DataFrame(results)
results_df.to_csv('cross_project_prediction_results.csv', index_label='Training Project')

print("Cross-project prediction results saved to 'cross_project_prediction_results.csv'.")
