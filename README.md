### Enhanced AI-Powered Network Intrusion Detection System

# Project Overview:

This project focuses on building a robust, AI-powered intrusion detection system (IDS) using machine learning and deep learning techniques. Models were trained and evaluated on the UNSW-NB15 dataset, covering both traditional machine learning models (Random Forest, XGBoost) and a deep learning model (Feedforward Neural Network, FNN).

# Dataset Summary:
Dataset: UNSW-NB15

Description: The dataset contains a comprehensive set of normal and malicious network flows captured in a realistic environment.

Key Features:
> 49 features (after preprocessing 42 numeric features were used)

> Binary classification: Normal vs. Attack

> Realistic modern attack types including DoS, Analysis, Exploits, etc.

# Preprocessing:

> Handled missing values in ct_flw_http_mthd, is_ftp_login, and attack_cat.

> Removed duplicate rows (480,632 duplicates).

Label encoding for categorical features.

> Standardization (Z-score normalization) for numerical stability.

# Model Training
# Random Forest:

Untuned and hyperparameter-tuned versions.

# XGBoost:

Untuned and hyperparameter-tuned versions.

# Feedforward Neural Network (FNN):

Designed using Keras.

Hyperparameter tuning via Keras Tuner - Hyperband.

# Evaluation Metrics:

Accuracy, Precision, Recall, F1-Score, ROC-AUC.

# Visualizations:

Confusion matrices, Precision-Recall curves, Histograms, Training/Validation curves.

# Main Libraries Used:

TensorFlow / Keras

Scikit-learn

XGBoost

Keras Tuner

Matplotlib, Seaborn

Pandas, NumPy

# Privacy and Ethical Considerations

The UNSW-NB15 dataset is publicly available and openly licensed for academic research purposes. The dataset was generated in a controlled space in a research lab. It does not contain any personally identifiable information (PII) and is anonymized. The dataset is solely used for research purposes. This project will not generate real attacks on the live network traffic. As the dataset is compliant with data protection laws such as GDPR, so there are no ethical concerns, and the dataset fulfils the ethical requirements of UH ethical policies. 
