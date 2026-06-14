# Cardiac Arrhythmia Detection using Machine Learning

A machine learning project to detect cardiac arrhythmia from ECG signals using the MIT-BIH Arrhythmia Dataset.

---

## Problem Statement

Cardiac arrhythmia is an irregular heartbeat that can be life-threatening if not detected early. Manual analysis of ECG signals by doctors is time-consuming and prone to error. This project builds a machine learning model that automatically classifies ECG heartbeat segments as **Normal** or **Abnormal**.

---

## Dataset

- **Source:** [MIT-BIH Arrhythmia Dataset](https://www.kaggle.com/datasets/shayanfazeli/heartbeat)
- Each row represents one heartbeat segment with **187 time-step features** and a label
- **Labels:**
  - 0 → Normal
  - 1 → Supraventricular Premature Beat
  - 2 → Premature Ventricular Contraction
  - 3 → Fusion Beat
  - 4 → Unclassifiable Beat

---

## Project Structure

```
cardiac-arrhythmia-detection/
│
├── data/                        # Raw and processed datasets
├── notebooks/
│   ├── 01_exploration.ipynb     # EDA and class distribution
│   ├── 02_preprocessing.ipynb  # Balancing, splitting, scaling
│   ├── 03_modeling.ipynb        # Model training
│   └── 04_evaluation.ipynb      # Model evaluation and comparison
├── src/
│   ├── preprocess.py            # Preprocessing functions
│   ├── model.py                 # Model training functions
│   └── evaluate.py              # Evaluation functions
├── results/                     # Saved plots and outputs
├── README.md
└── requirements.txt
```

---

## Approach

### 1. Exploratory Data Analysis
- Loaded train and test CSV files
- Analyzed class distribution — dataset found to be heavily imbalanced (83% Normal)

### 2. Preprocessing
- Converted 5-class problem to **binary classification** (Normal vs Abnormal)
- Applied **oversampling (resample)** to balance the training data
- Applied **StandardScaler** to normalize features

### 3. Modeling
Trained and compared two models:
- Logistic Regression
- K-Nearest Neighbors (KNN)

### 4. Evaluation
Evaluated using accuracy, confusion matrix, precision, recall, and F1-score.

---

## Results

| Model | Accuracy | F1 (Normal) | F1 (Abnormal) |
|-------|----------|-------------|---------------|
| Logistic Regression | 84.1% | 0.90 | 0.63 |
| KNN | **97.0%** | **0.98** | **0.92** |

**KNN outperformed Logistic Regression** across all metrics, achieving 97% accuracy and 0.92 F1-score on abnormal heartbeat detection.

---

## How to Run

1. Clone the repository:
```bash
git clone https://github.com/yourusername/cardiac-arrhythmia-detection.git
cd cardiac-arrhythmia-detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/shayanfazeli/heartbeat) and place CSV files in `data/`

4. Run notebooks in order:
```
01_exploration → 02_preprocessing → 03_modeling → 04_evaluation
```

---

## Tech Stack

- Python
- NumPy, Pandas
- Scikit-learn
- Matplotlib, Seaborn

---

## Key Learnings

- Handling class imbalance with oversampling
- Importance of proper train/test scaling to avoid data leakage
- Binary classification for medical signal data
- Model comparison using F1-score over accuracy for imbalanced datasets
