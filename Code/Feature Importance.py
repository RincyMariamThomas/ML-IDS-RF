import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from tabulate import tabulate

# Load the dataset
df = pd.read_csv(r'C:\Users\rincy\Desktop\ML_IDS - RFC\Dataset\Train\KDDTrain.csv')

# Data Exploration
print("Shape of the dataset:", df.shape)
print("Columns in the dataset:", df.columns)
print("Class distribution:\n", df['class'].value_counts())

# Data preprocessing
# Encode categorical variables using LabelEncoder
label_encoders = {}
for column in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Splitting the dataset into train and test sets
X = df.drop('class', axis=1)
y = df['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model training without data reduction
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train_scaled, y_train)

# Predictions
y_pred = rf_classifier.predict(X_test_scaled)

# Model evaluation
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix in tabular form
conf_matrix = confusion_matrix(y_test, y_pred)
conf_matrix_table = tabulate(conf_matrix, headers=['Predicted ' + str(i) for i in range(conf_matrix.shape[1])],
                             showindex=['Actual ' + str(i) for i in range(conf_matrix.shape[0])], tablefmt='pretty')
print("Confusion Matrix:")
print(conf_matrix_table)

# Feature Importance
feature_importance = rf_classifier.feature_importances_
sorted_idx = np.argsort(feature_importance)[::-1]
features = X.columns

# Visualization of feature importance
plt.figure(figsize=(10, 6))
plt.bar(range(X.shape[1]), feature_importance[sorted_idx], align='center')
plt.xticks(range(X.shape[1]), [features[i] for i in sorted_idx], rotation=90)
plt.xlabel('Features')
plt.ylabel('Importance Score')
plt.title('Feature Importance')
plt.show()
