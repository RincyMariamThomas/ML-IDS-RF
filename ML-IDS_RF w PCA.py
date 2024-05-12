import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tabulate import tabulate

def preprocess_data(data):
    """
    Preprocess the dataset:
    - Convert the 'class' column to binary
    - Encode categorical 'protocol' column
    """
    # Convert the 'class' column to binary
    data["class"] = data["class"].apply(lambda x: 0 if x == "normal" else 1)
    
    # Select relevant columns
    features = data[["src_bytes", "dst_bytes", "protocol"]].copy()  # Make a copy to avoid the warning
    labels = data["class"]
    
    # Label encoding for 'protocol' column
    label_encoder = LabelEncoder()
    features.loc[:, 'protocol'] = label_encoder.fit_transform(features['protocol'])
    
    return features, labels

def train_model(X_train, y_train, param_grid):
    """
    Train the model using Random Forest with GridSearchCV for hyperparameter tuning.
    """
    # Splitting the dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    # Standard scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Apply PCA
    pca = PCA(n_components=2)  # Choose number of components as per requirement
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    # Initialize Random Forest classifier
    rf_classifier = RandomForestClassifier(random_state=42)
    
    # Initialize GridSearchCV
    grid_search = GridSearchCV(estimator=rf_classifier, param_grid=param_grid,
                               cv=3, n_jobs=-1, verbose=2, scoring='accuracy')
    
    # Perform grid search
    grid_search.fit(X_train_pca, y_train)
    
    # Get best hyperparameters
    best_params = grid_search.best_params_
    
    return grid_search, X_test_pca, y_test, best_params

def evaluate_model(grid_search, X_test_pca, y_test):
    """
    Evaluate the model's performance and print out the results.
    """
    # Make predictions
    y_pred_test_best = grid_search.best_estimator_.predict(X_test_pca)
    
    # Performance evaluation
    test_accuracy_best = accuracy_score(y_test, y_pred_test_best)
    
    # Confusion matrix
    conf_matrix_best = confusion_matrix(y_test, y_pred_test_best)
    
    # Print Confusion Matrix
    print("Confusion Matrix:")
    print(tabulate(conf_matrix_best, headers=['Predicted 0', 'Predicted 1'],
                   showindex=['Actual 0', 'Actual 1'], tablefmt='pretty'))
    
    # Print classification report
    print("Classification Report:")
    print(classification_report(y_test, y_pred_test_best))
    
    # Print performance metrics and best hyperparameters
    print("Best Hyperparameters:", grid_search.best_params_)
    print("Test Accuracy (with PCA) with Best Hyperparameters:", test_accuracy_best)
    
    # Plotting the confusion matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(conf_matrix_best, cmap='Blues', interpolation='nearest')
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()
    
    # ROC Curve
    y_pred_prob = grid_search.best_estimator_.predict_proba(X_test_pca)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()
    
    # AUC Score
    auc_score = roc_auc_score(y_test, y_pred_prob)
    print("AUC Score:", auc_score)

    # Calculating computational complexity for training
    train_complexity = grid_search.cv_results_['mean_fit_time'].mean()
    print("Average Training Complexity (seconds):", train_complexity)
    
    # Calculating computational complexity for running
    run_complexity = grid_search.cv_results_['mean_score_time'].mean()
    print("Average Running Complexity (seconds):", run_complexity)

def display_data_info(data):
    """
    Display information about the dataset.
    """
    # Get the shape of the dataset
    num_rows, num_cols = data.shape
    
    # Display the number of data points
    print("Number of data points:", num_rows)
    
    # Display column names and their data types
    print("\nColumn names and data types:")
    print(data.dtypes)
    
    # Display first few rows of the dataset
    print("\nFirst few rows of the dataset:")
    print(data.head())

def main(dataset_path):
    """
    Main function to load the dataset, preprocess data, train the model, and evaluate its performance.
    """
    # Load the dataset
    data = pd.read_csv(dataset_path)
    
    # Display information about the dataset
    display_data_info(data)
    
    # Preprocess data
    features, labels = preprocess_data(data)
    
    # Display number of data points and feature dimensionality
    print("\nNumber of data points:", len(features))
    print("Number of features used:", len(features.columns))
    
    # Display the data points
    print("\nData points:")
    print(features)
    
    # Define hyperparameters grid
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    
    # Train model
    grid_search, X_test_pca, y_test, best_params = train_model(features, labels, param_grid)
    
    # Evaluate model
    evaluate_model(grid_search, X_test_pca, y_test)

if __name__ == "__main__":
    # Path to the dataset
    dataset_path = r'C:\Users\rincy\Desktop\ML_IDS - RFC\Dataset\Train\KDDTrain.csv'
    
    # Run the main function
    main(dataset_path)