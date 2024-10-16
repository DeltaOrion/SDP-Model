import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# Load dataset
data = pd.read_csv('../../../hand-data/method-p.csv')

# Convert 'Number of Bugs' to binary classification (>= 1 becomes 1, else 0)
data['Number of Bugs'] = (data['Number of Bugs'] >= 1).astype(int)

# Preprocess data
X = data.drop(columns=['Project', 'Hash', 'LongName', 'Parent', 'Number of Bugs'])
y = data['Number of Bugs']

# Split the data into train, validation, and test sets (80/10/10)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Initialize RandomForest model
model = RandomForestClassifier(random_state=42, class_weight='balanced')

# Fit model to the training data
model.fit(X_train, y_train)

# Save the trained model, scaler, and test set for future evaluation
joblib.dump(model, 'random_forest_model.joblib')
joblib.dump(scaler, 'scaler.joblib')
test_data = {"X_test": X_test, "y_test": y_test}
joblib.dump(test_data, 'test_data.joblib')  # Save test data separately

print("Model, scaler, and test data saved.")
