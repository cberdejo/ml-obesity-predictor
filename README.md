# Obesity Classification Models with scikit-learn

[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.2-orange?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)


This project compares different machine learning classifiers using the obesity dataset. The goal is to evaluate how well each model predicts obesity levels based on lifestyle and health-related features.

All models and techniques were implemented using the **[scikit-learn](https://scikit-learn.org/)** library.

## 📘 Models Used

### 1. K-Nearest Neighbors (KNN)
- Classifies each instance based on the majority label of its *k* nearest neighbors in feature space.
- Key hyperparameters:
  - `n_neighbors`
  - `weights` (`uniform`, `distance`)
  - `metric` (`euclidean`, `manhattan`, `cosine`)

### 2. Decision Tree Classifier
- Builds a tree structure to split the data based on feature values.
- Easily interpretable and handles both numerical and categorical data.
- Key hyperparameters:
  - `criterion` (`gini`, `entropy`, `log_loss`)
  - `max_depth`
  - `min_samples_split`

### 3. Support Vector Classifier (SVC)
- Finds the optimal hyperplane to separate classes with maximum margin.
- Performs well in high-dimensional spaces.
- Key hyperparameters:
  - `C`: regularization
  - `kernel`: `linear`, `rbf`, `sigmoid`
  - `gamma`, `degree`, `coef0`

> ⚠️ Note: `SVC` does not output probabilities by default. For **soft voting**, probability estimation was enabled using `SVC(probability=True)` or via `CalibratedClassifierCV` for efficiency.

### 4. Voting Classifier (Ensemble)
- Combines predictions from multiple models:
  - **Hard Voting**: majority rule
  - **Soft Voting**: averages class probabilities
- Improves performance by leveraging the strengths of individual models.

## 🧪 Methodology

- Data preprocessing with `StandardScaler`
- Hyperparameter tuning via `GridSearchCV`
- Cross-validation with both **K-Fold** and **Leave-One-Out**
- Model evaluation using:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - Confusion matrix



## 📦 Dependencies

Install required packages using:

```bash
pip install -r requirements.txt 
```

## 🚀 How to Use
Clone this repository.

Open the notebook `classification_analysis.ipynb.`

Run all cells to train, evaluate, and compare classifiers.

🧠 Powered by scikit-learn · Built with ❤️ for machine learning exploration.
