# Applying Machine Learning for Intrusion Detection in Network Security using Random Forest Algorithm

## Introduction
Intrusion detection plays a crucial role in maintaining network security by identifying malicious activities. The objective of this project is to apply machine learning techniques, specifically the Random Forest algorithm, for intrusion detection. By leveraging the NSL-KDD dataset, the aim is to develop a model capable of accurately classifying network traffic as normal or intrusive.

## Project Overview
- **Dataset**: [NSL-KDD](https://www.kaggle.com/datasets/hassan06/nslkdd/data)
- **Total Columns**: 43
- **Number of Data Points**: 125,973
- **Number of Features Used**: 3

### Column Names and Data Types
- `duration`: int64
- `protocol`: object
- `service`: object
- `flag`: object
- `src_bytes`: int64
- `dst_bytes`: int64
- `land`: int64
- `wrong_fragment`: int64
- `urgent`: int64
- `hot`: int64
- `num_failed_logins`: int64
- `logged_in`: int64
- `num_compromised`: int64
- `root_shell`: int64
- `su_attempted`: int64
- `num_root`: int64
- `num_file_creations`: int64
- `num_shells`: int64
- `num_access_files`: int64
- `num_outbound_cmds`: int64
- `is_host_login`: int64
- `is_guest_login`: int64
- `count`: int64
- `srv_count`: int64
- `serror_rate`: float64
- `srv_serror_rate`: float64
- `rerror_rate`: float64
- `srv_rerror_rate`: float64
- `same_srv_rate`: float64
- `diff_srv_rate`: float64
- `srv_diff_host_rate`: float64
- `dst_host_count`: int64
- `dst_host_srv_count`: int64
- `dst_host_same_srv_rate`: float64
- `dst_host_diff_srv_rate`: float64
- `dst_host_same_src_port_rate`: float64
- `dst_host_srv_diff_host_rate`: float64
- `dst_host_serror_rate`: float64
- `dst_host_srv_serror_rate`: float64
- `dst_host_rerror_rate`: float64
- `dst_host_srv_rerror_rate`: float64
- `class`: object
- `level`: int64

### Consistent Feature Selection
I ran a code [Feature Importance.py](https://github.com/RincyMariamThomas/ML-IDS-RF/blob/main/Code/Feature%20Importance.py) to plot a "Feature Importance graph". Below is the output. 
![Feature Importance Plot](https://github.com/RincyMariamThomas/ML-IDS-RF/blob/main/Output/RF%20Feature%20Importance/Feature%20Importance.png)

I chose the following attributes which work well with other algorithms([KNN](https://github.com/akshaygk20/IDS-using-KNN), K-Means & [Isolation Forest](https://github.com/AyaanJahanzebAhmed/network_security_project)) and are important features.
- **Features**:
  - `src_bytes`: Amount of data sent from the source to the destination.
  - `dst_bytes`: Amount of data sent from the destination back to the source.
  - `protocol`: Communication protocol used (`tcp`, `udp`, `icmp`).
- **Target Variable**:
  - `class`: Identifies the class of network traffic as either 'normal' or 'anomaly'.

### Purpose
- Ensure uniformity in input data among all projects for meaningful comparisons.
- Facilitate the evaluation of machine learning algorithms' performance.
- Evaluate the performance of Random Forest and compare results with other algorithms like Isolation Forest, KNN, and K-Means.

## What is Random Forest and Why Use It?
Random Forest is an ensemble learning method that constructs a multitude of decision trees during training and outputs the mode of the classes as the prediction. By employing an ensemble learning approach, this project effectively reduces overfitting and enhances generalization by aggregating multiple decision trees' predictions.

### Reasons for Using Random Forest
- **Robustness**: Handles high-dimensional data with categorical and numerical features effectively.
- **Flexibility**: Suitable for both classification and regression tasks.
- **Versatility and Performance**: Effective in detecting and classifying different types of network intrusions.

## How Random Forest Works
The Random Forest algorithm is a popular ensemble learning method used for both classification and regression tasks. Here's how it works:

1. **Bootstrapping (Random Sampling with Replacement)**:
   - Random Forest creates multiple bootstrap samples from the original dataset. Each sample may contain duplicate instances.

2. **Building Decision Trees**:
   - For each bootstrap sample, a decision tree is grown. At each node, only a random subset of features is considered for splitting.

3. **Voting (Classification) or Averaging (Regression)**:
   - The algorithm combines the predictions of all trees. For classification, it uses majority voting. For regression, it averages the predictions.

### Key Characteristics of Random Forest
- **Ensemble Learning**: Combines multiple decision trees to improve predictive performance and robustness.
- **Bagging (Bootstrap Aggregating)**: Uses bootstrap sampling to create diverse training datasets.
- **Random Feature Selection**: Adds diversity by considering only a random subset of features at each node.
- **Robustness**: Reduces overfitting and noise in the data.
- **Scalability**: Handles large datasets with high dimensionality efficiently.

## How Random Forest Works for My Code
### Attributes: ["src_bytes", "dst_bytes", "protocol"]
These attributes are used to form decision boundaries within each decision tree.

### Preprocessing
- **Categorical Attribute Encoding**:
  - The `protocol` attribute is label-encoded to convert it into numerical form.
- **Scaling**:
  - The features are then scaled using `StandardScaler` to normalize the data.

### Random Subsets
Each decision tree uses a different random subset of these features and data points to train on. This ensures diversity among the trees.

### Node Splits
- At each node in a decision tree, a subset of the features (e.g., `src_bytes` and `protocol` in one tree, `dst_bytes` and `src_bytes` in another) is randomly selected.
- The best split is chosen based on the criterion (like Gini impurity). For example, if `src_bytes < 200` might be a decision point at a node.

### Leaf Nodes
Once a leaf node is reached, it represents a class label based on the majority class of the samples that reached that node.

### Making a Prediction
- For a new data point, the data is passed down each tree, following the decision rules at each node until it reaches a leaf node.
- Each tree makes its prediction.
- For classification, the final class is determined by majority voting across all trees.

## Hyperparameters
The provided explanation outlines the meaning and significance of each hyperparameter in the Random Forest algorithm and the corresponding choices specified in the param_grid dictionary for hyperparameter tuning using GridSearchCV. Let's break it down:

1. **n_estimators**:
   - *Definition*: This parameter specifies the number of trees in the Random Forest ensemble.
   - *Significance*: Increasing the number of trees typically improves the model's performance by reducing variance and increasing robustness. More trees allow the model to capture more complex relationships in the data and provide more stable predictions.
   - *Choices*: The param_grid specifies two values for n_estimators: 50 and 100. These values represent different sizes of the forest, allowing GridSearchCV to evaluate the model's performance with different numbers of trees.

2. **max_depth**:
   - *Definition*: This parameter controls the maximum depth of each decision tree in the Random Forest.
   - *Significance*: Limiting the maximum depth of the trees helps prevent overfitting by restricting the complexity of individual trees. Shallower trees are less likely to capture noise or irrelevant patterns in the data, leading to better generalization on unseen data.
   - *Choices*: The param_grid specifies two values for max_depth: 10 and 20. These values represent different levels of tree depth, allowing GridSearchCV to evaluate the model's performance with different levels of tree complexity.

3. **min_samples_split and min_samples_leaf**:
   - *Definition*: These parameters control the minimum number of samples required to split an internal node (min_samples_split) and the minimum number of samples required to be at a leaf node (min_samples_leaf).
   - *Significance*: By imposing minimum sample requirements, these parameters help prevent overfitting by controlling the growth of individual trees. They ensure that nodes are not split if they contain a small number of samples, thus promoting simpler and more generalized trees.
   - *Choices*: The param_grid specifies two values for min_samples_split and min_samples_leaf: [2, 5] and [1, 2], respectively. These values represent different thresholds for node splitting and leaf formation, allowing GridSearchCV to evaluate the model's performance with different levels of regularization.

In summary, the param_grid dictionary provides a range of values for each hyperparameter, allowing GridSearchCV to systematically search for the optimal combination of hyperparameters that maximizes the Random Forest model's performance for intrusion detection in network security. Each hyperparameter plays a crucial role in controlling the complexity, robustness, and generalization ability of the model, and tuning them appropriately is essential for building an effective and reliable intrusion detection system.

## Libraries used in my code
- Pandas is used for data manipulation
- Sklearn for machine learning tasks
- Matplotlib for plotting
- Tabulate for tabular representation.

## Using PCA in the Code
Principal Component Analysis (PCA) is a powerful technique for dimensionality reduction that transforms the original features into a new set of features, the principal components, which are orthogonal and capture the maximum variance in the data. Here's a detailed explanation of how PCA is being applied to the code, how it changes the working of the code, and what happens to the three attributes (`src_bytes`, `dst_bytes`, `protocol`).

#### Step-by-Step Application of PCA in the Code

1. **Preprocessing**:
   - **Label Encoding**: The `protocol` attribute, which is categorical, is converted into numerical form using label encoding.
   - **Scaling**: All features (`src_bytes`, `dst_bytes`, `protocol`) are scaled using `StandardScaler` to ensure that they have a mean of 0 and a standard deviation of 1. This step is crucial for PCA, which is sensitive to the scale of the data.

2. **Applying PCA**:
   - **Initialization**: PCA is initialized with `n_components=2`, which means we want to reduce the three original features to two principal components.
   - **Fitting and Transforming**: The scaled data is then fit and transformed using PCA. This involves calculating the covariance matrix of the features, determining the eigenvalues and eigenvectors, and projecting the data onto the new principal component axes.

3. **Training the Model**:
   - The transformed data, now in the form of two principal components, is used to train the Random Forest classifier.
   - The training process involves the same steps as without PCA, but instead of the original three features, the model now works with the two principal components.

#### Detailed Changes in the Working of the Code

1. **Dimensionality Reduction**:
   - **From 3 to 2 Dimensions**: Originally, the dataset had three features (`src_bytes`, `dst_bytes`, `protocol`). After applying PCA, these three features are transformed into two new features (principal components). These principal components are linear combinations of the original features and capture the maximum variance in the data.

2. **Interpretation of Principal Components**:
   - The principal components are new axes that are orthogonal (uncorrelated) and represent the directions of maximum variance in the data. Each principal component is a weighted sum of the original features. For example, the first principal component might be a combination of `src_bytes`, `dst_bytes`, and `protocol`, with specific weights indicating their contribution.

3. **Impact on Model Training**:
   - **Simplicity and Speed**: With fewer dimensions, the model training becomes faster and simpler. The Random Forest algorithm now has fewer features to consider at each node split, which can speed up the training process.
   - **Potential Information Loss**: While PCA captures the most important variance, it might lose some information present in the original features. However, by retaining the components that capture the most variance, we aim to minimize this loss.

4. **Model Performance**:
   - **Effectiveness**: PCA can help in reducing noise and irrelevant information, potentially leading to better generalization and performance on unseen data.
   - **Evaluation Metrics**: The performance metrics (accuracy, precision, recall, etc.) may change slightly due to the dimensionality reduction. It's important to compare these metrics with and without PCA to assess the impact.

#### What Happens to the Three Attributes (`src_bytes`, `dst_bytes`, `protocol`)

1. **Original Attributes**:
   - `src_bytes`: Amount of data sent from the source to the destination.
   - `dst_bytes`: Amount of data sent from the destination back to the source.
   - `protocol`: Communication protocol used (`tcp`, `udp`, `icmp`), label-encoded.

2. **After Applying PCA**:
   - The original attributes are transformed into two new attributes (principal components). These components are combinations of `src_bytes`, `dst_bytes`, and `protocol`.
   - For instance, if the principal components are denoted as PC1 and PC2:
     - PC1 might be a combination like \(0.5 \times \text{src_bytes} + 0.3 \times \text{dst_bytes} - 0.2 \times \text{protocol}\)
     - PC2 might be another combination like \(-0.4 \times \text{src_bytes} + 0.6 \times \text{dst_bytes} + 0.1 \times \text{protocol}\)
   - The exact coefficients (weights) are determined by the PCA process and indicate the contribution of each original feature to the principal components.

By applying PCA with `n_components=2`, we transform the original three features into two principal components that capture the most significant variance in the data. This transformation simplifies the dataset, potentially improves the performance and speed of the Random Forest model, and helps in mitigating the curse of dimensionality. The transformed features are combinations of the original attributes, and the model operates on these new features instead of the original ones. This approach highlights the balance between reducing dimensionality for simplicity and retaining enough information for accurate classification, making it a valuable preprocessing step in machine learning pipelines.

## Methodology
### Without PCA
1. Preprocess the dataset by encoding categorical variables and scaling numerical features.
2. Train a Random Forest classifier using GridSearchCV for hyperparameter tuning.
3. Evaluate model performance using accuracy, confusion matrix, and classification report.

### Results - Without PCA
- **Confusion Matrix**:
  ```plaintext
   +----------+-------------+-------------+
   |          | Predicted 0 | Predicted 1 |
   +----------+-------------+-------------+
   | Actual 0 |    12771    |     651     |
   | Actual 1 |     23      |    11750    |
   +----------+-------------+-------------+
  ```
- **Classification Report**:
  ```plaintext
               precision    recall  f1-score   support
            0       1.00      0.95      0.97     13422
            1       0.95      1.00      0.97     11773
     accuracy                           0.97     25195
    macro avg       0.97      0.97      0.97     25195
  weighted avg       0.97      0.97      0.97     25195
  ```
- **Best Hyperparameters**: {'max_depth': 20, 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 100}
- **Test Accuracy**: 0.9732486604485017

### With PCA
1. Implement Principal Component Analysis (PCA) for dimensionality reduction.
2. Transform the dataset into a lower-dimensional space using PCA.
3. Train a Random Forest classifier on the reduced dataset.

### Results - With PCA
- **Confusion Matrix**:
  ```plaintext
   +----------+-------------+-------------+
   |          | Predicted 0 | Predicted 1 |
   +----------+-------------+-------------+
   | Actual 0 |    12698    |     724     |
   | Actual 1 |     45      |    11728    |
   +----------+-------------+-------------+
  ```
- **Classification Report**:
  ```plaintext
               precision    recall  f1-score   support
            0       1.00      0.95      0.97     13422
            1       0.94      1.00      0.97     11773
     accuracy                           0.97     25195
    macro avg       0.97      0.97      0.97     25195
  weighted avg       0.97      0.97      0.97     25195
  ```
- **Best Hyperparameters**: {'max_depth': 20, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 50}
- **Test Accuracy**: 0.9694780710458424

### ROC, AUC and Complexity
- **Without PCA**:
- ![ROC Curve without PCA](https://github.com/RincyMariamThomas/ML-IDS-RF/blob/main/Output/RF%20without%20PCA/ROC%20wo%20PCA.png)
  - **AUC Score**: 0.9810941505952206
  - **Training Complexity**: ~4.97 seconds
  - **Running Complexity**: ~0.23 seconds
- **With PCA**:
- ![ROC Curve with PCA](https://github.com/RincyMariamThomas/ML-IDS-RF/blob/main/Output/RF%20with%20PCA/ROC%20w%20PCA.png)
  - **AUC Score**: 0.979578749164822
  - **Training Complexity**: ~5.66 seconds
  - **Running Complexity**: ~0.26 seconds

## Performace Metrics Comparision of KNN, K-Means, Random Forest & Isolation Forest
Link will be added soon.

## Potential Improvements
- **Feature Engineering**:
  - Explore additional or engineered features to enhance model performance.
  - Experiment with different feature selection techniques.
- **Hyperparameter Optimization**:
  - Fine-tune hyperparameters further.
  - Utilize advanced optimization techniques like Bayesian optimization.
- **Ensemble Methods**:
  - Investigate other ensemble methods like stacking or boosting.
- **Model Interpretability**:
  - Implement techniques for model interpretability (e.g., SHAP values).

## Conclusion
Random Forest demonstrates promising results for intrusion detection in network security. Leveraging machine learning techniques enhances our ability to detect and mitigate network threats. Continued research and experimentation are essential for advancing intrusion detection systems.

## References
- [NSL-KDD Dataset](https://www.kaggle.com/datasets/hassan06/nslkdd/data)
- [GitHub Repository](https://github.com/RincyMariamThomas/ML-IDS-RF)
- Research Papers:
  - [IEEE Paper](https://ieeexplore.ieee.org/abstract/document/9778286)
  - [Springer Article](https://link.springer.com/article/10.1007/s00500-021-05893-0)
