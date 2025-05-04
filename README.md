# 💳 Credit Card Fraud Detection

This project focuses on detecting fraudulent credit card transactions using various machine learning models and oversampling techniques like SMOTE to handle imbalanced data.

---

## 📂 Dataset

- **Source**: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Description**: Contains transactions made by European cardholders in September 2013.
- **Shape**: 284,807 rows × 31 columns
- **Class Distribution**: 
  - Legitimate (0): 284,315
  - Fraudulent (1): 492

---

## ✅ Objective

To build a machine learning pipeline that:
- Detects fraudulent transactions with high precision and recall.
- Handles extreme class imbalance effectively using SMOTE.

---

## ⚙️ Technologies Used

- Python (Pandas, NumPy)
- Scikit-learn
- imbalanced-learn (SMOTE)
- Google Colab (for execution)

---

## 🧠 ML Models Implemented

- Logistic Regression
- Random Forest Classifier
- Decision Tree Classifier
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Gaussian Naive Bayes

---

## 🧪 Evaluation Metrics

- **Confusion Matrix**
- **Classification Report** (Precision, Recall, F1-Score)
- **Cross-Validation Scores**


---

## 🔄 Data Processing Pipeline

1. Load dataset from Google Drive
2. Normalize features using `StandardScaler`
3. Handle class imbalance using `SMOTE`
4. Split data into train and test sets
5. Train multiple classifiers and compare results
6. Evaluate performance using standard classification metrics

---

## 📊 Results

The best models were evaluated based on recall of the positive (fraudulent) class to reduce the chances of missing fraud:

| Model                | Recall | Precision | F1-Score |
|---------------------|--------|-----------|----------|
| Logistic Regression |   --   |     --    |    --    |
| Random Forest       |   --   |     --    |    --    |
| Decision Tree       |   --   |     --    |    --    |

*To be filled after model evaluation.*

---

## 📎 How to Run

1. Upload the dataset to your Google Drive.
2. Mount Google Drive in Google Colab:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
