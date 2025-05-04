# 💳 Credit Card Fraud Detection

This project applies various machine learning algorithms to detect fraudulent credit card transactions. Given the **highly imbalanced** nature of fraud detection datasets, techniques like **SMOTE (Synthetic Minority Oversampling Technique)** have been used to balance the classes.

---

## 📂 Dataset

- **Source**: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Description**: 
  - Transactions made by European cardholders in September 2013.
  - Contains **284,807** transactions, with only **492 fraud cases** (Class 1).
  - Features are anonymized with PCA (V1-V28), along with `Time`, `Amount`, and `Class`.

---

## 🧠 Algorithms Used

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Naive Bayes (GaussianNB)
- K-Nearest Neighbors
- Support Vector Classifier (SVC)

---

## ⚙️ Workflow

1. **Import Data**
2. **Data Preprocessing**
   - Drop/Handle Nulls (if any)
   - Feature Scaling using `StandardScaler`
3. **Train-Test Split**
4. **Handling Imbalanced Data**
   - Apply **SMOTE** to oversample the minority (fraud) class
5. **Model Training**
   - Train models on balanced data
6. **Evaluation**
   - `Classification Report`
   - `Confusion Matrix`
   - Cross-validation scores

---

## 📈 Performance Metrics

Evaluation is done using:
- **Accuracy**
- **Precision**
- **Recall** (Important for fraud detection)
- **F1-Score**


---

## 🔬 Key Learnings

- Importance of handling **imbalanced datasets** in classification problems.
- SMOTE helps improve minority class prediction without simply duplicating data.
- Comparative analysis of multiple classifiers shows trade-offs between precision and recall.

---

## 🛠 Requirements

Install dependencies using pip:

```bash
pip install pandas scikit-learn imbalanced-learn
```


## Run on Google Colab
This project is Colab-compatible. To access the dataset stored in Google Drive:
from google.colab import drive
drive.mount('/content/drive')

## 📬 Contact
Author: Sandeep
Feel free to connect on LinkedIn or explore other projects.
linkedin : [https://www.linkedin.com/in/sandeep-chopra-8ba2b7272]

## 📜 License
This project is open-source and available under the MIT License.



