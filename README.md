# Applying Machine Learning for Intrusion Detection in Network Security using Random Forest Algorithm

## Introduction
This project focuses on using machine learning techniques for network intrusion detection, with a specific emphasis on the Random Forest algorithm. The goal is to develop a model that can effectively classify network traffic as either normal or intrusive using the NSL-KDD dataset.

## Project Overview
- **Dataset**: [NSL-KDD](https://www.kaggle.com/datasets/hassan06/nslkdd/data)
- **Total Columns**: 43
- **Number of Data Points**: 125,973
- **Features Used**: 3

### Features
- `src_bytes`: Data sent from the source to the destination.
- `dst_bytes`: Data sent from the destination back to the source.
- `protocol`: Communication protocol used (`tcp`, `udp`, `icmp`).

### Target Variable
- `class`: Indicates whether the network traffic is 'normal' or 'anomaly'.

## Purpose
- Standardize input data across different projects for effective comparison.
- Assess the performance of the Random Forest algorithm in comparison with other machine learning algorithms like Isolation Forest, KNN, and K-Means.

## Random Forest Algorithm
Random Forest is an ensemble learning method that combines multiple decision trees to enhance predictive accuracy and control overfitting. 

### Key Characteristics
- **Ensemble Learning**: Aggregates predictions from multiple trees.
- **Bagging**: Uses bootstrap sampling for diverse training datasets.
- **Random Feature Selection**: Considers a random subset of features for splits.
- **Robustness**: Reduces overfitting and handles high-dimensional data effectively.

## Implementation Details
### Without PCA
- **Preprocessing**: Encode categorical variables and scale features.
- **Model Training**: Train a Random Forest classifier using GridSearchCV for hyperparameter tuning.
- **Results**: Achieved an accuracy of 97.32% with the best hyperparameters.

### With PCA
- **Preprocessing**: Apply PCA to reduce dimensionality from 3 to 2 components.
- **Model Training**: Train the Random Forest classifier on the reduced dataset.
- **Results**: Achieved an accuracy of 96.95% with the best hyperparameters.

### ROC Curve
- **Without PCA**: [ROC Curve without PCA](https://github.com/RincyMariamThomas/ML-IDS-RF/blob/main/Output/RF%20without%20PCA/ROC%20wo%20PCA.png)
- **With PCA**: [ROC Curve with PCA](https://github.com/RincyMariamThomas/ML-IDS-RF/blob/main/Output/RF%20with%20PCA/ROC%20w%20PCA.png)

## Comparison with Other Algorithms
- **Isolation Forest**: [GitHub Repository](https://github.com/AyaanJahanzebAhmed/network_security_project)
- **KNN**: [GitHub Repository](https://github.com/akshaygk20/IDS-using-KNN)
- **K-Means**: [GitHub Repository](https://github.com/swethabotta/IDS_Using_KMEANS)

## Potential Improvements
- **Feature Engineering**: Explore additional features or techniques.
- **Hyperparameter Optimization**: Utilize advanced techniques like Bayesian optimization.
- **Ensemble Methods**: Investigate methods like stacking or boosting.
- **Model Interpretability**: Implement interpretability techniques (e.g., SHAP values).

## Conclusion
The Random Forest algorithm demonstrates strong performance in detecting network intrusions, showcasing its effectiveness in improving network security. Ongoing research and experimentation are vital for enhancing intrusion detection systems.

## References
- [NSL-KDD Dataset](https://www.kaggle.com/datasets/hassan06/nslkdd/data)
- [GitHub Repository](https://github.com/RincyMariamThomas/ML-IDS-RF)
- Research Papers:
  - [IEEE Paper](https://ieeexplore.ieee.org/abstract/document/9778286)
  - [Springer Article](https://link.springer.com/article/10.1007/s00500-021-05893-0)
