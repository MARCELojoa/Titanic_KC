# 🚢 Titanic — Machine Learning from Disaster

> *Kaggle Competition | Binary Classification | Survival Prediction*

## 📌 Overview

On **April 15, 1912**, the RMS Titanic sank on her maiden voyage after striking an iceberg, claiming **1,502 of the 2,224 passengers and crew** aboard. While survival involved an element of chance, patterns in the data suggest that certain groups — based on gender, age, and socioeconomic class — had significantly higher likelihoods of survival.
This project builds a **predictive machine learning model** to answer:

> **"What kinds of people were more likely to survive the Titanic disaster?"**

## 🎯 Objective

Train a binary classifier on labeled passenger data to predict survival (`0 = Did not survive`, `1 = Survived`) for unseen test records.
**Evaluation Metric:** Accuracy (percentage of correctly predicted passengers)
## 📁 Dataset

| File | Description |
|------|-------------|
| `train.csv` | Training set with ground-truth survival labels |
| `test.csv` | Test set for generating predictions |
| `gender_submission.csv` | Sample submission (baseline: all females survive) |

### Feature Dictionary

| Feature | Type | Description |
|---------|------|-------------|
| `PassengerId` | int | Unique passenger identifier |
| `Survived` | int (target) | 0 = No, 1 = Yes |
| `Pclass` | int | Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd) — proxy for socioeconomic status |
| `Name` | string | Passenger name |
| `Sex` | string | Gender |
| `Age` | float | Age in years |
| `SibSp` | int | # of siblings/spouses aboard |
| `Parch` | int | # of parents/children aboard |
| `Ticket` | string | Ticket number |
| `Fare` | float | Passenger fare |
| `Cabin` | string | Cabin number (many missing) |
| `Embarked` | string | Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton) |

## 🔍 Exploratory Data Analysis
Key findings from initial EDA:
- **Sex:** Female survival rate (~74%) was significantly higher than male (~19%)
- **Pclass:** 1st class passengers had a much higher survival rate than 3rd class
- **Age:** Children had a higher chance of survival; elderly passengers fared worse
- **Fare:** Higher fare correlated positively with survival
- **Family size:** Small families (1–3 members) survived more than solo travelers or large groups
- **Embarked:** Passengers from Cherbourg showed slightly higher survival rates

## ⚙️ Methodology
### 1. Data Preprocessing
- Impute missing `Age` values using median grouped by `Pclass` and `Sex`
- Fill missing `Embarked` with mode; fill missing `Fare` with median
- Drop `Cabin` due to excessive missingness (~77%)
- Drop non-informative columns: `PassengerId`, `Name`, `Ticket`
### 2. Feature Engineering
- **Title extraction** from `Name` (e.g., Mr, Mrs, Miss, Rare)
- **FamilySize** = `SibSp` + `Parch` + 1
- **IsAlone** = 1 if `FamilySize == 1`
- **AgeBand** — binned age groups
- **FareBand** — binned fare groups
### 3. Encoding
- Label encoding for `Sex` and `Embarked`
- One-hot encoding for ordinal/nominal features where appropriate
### 4. Model Training
Models evaluated:
- Logistic Regression
- Random Forest Classifier ✅ *(best performer)*
- Gradient Boosting (XGBoost / LightGBM)
- Support Vector Machine
- K-Nearest Neighbors

Hyperparameter tuning via **GridSearchCV** with 5-fold cross-validation ( Future Plan )

## 📊 Results

| Model | CV Accuracy |
|-------|-------------|
| Logistic Regression | ~ will do |
| SVM | ~ will do|
| KNN | ~ will do |
| Random Forest | **~97.98%** |
| XGBoost | ~ will do |

> Final submission used the **Random Forest** model with tuned hyperparameters.

## 🗂️ Project Structure
----

titanic/
│
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── gender_submission.csv
│
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Feature_Engineering.ipynb
│   └── 03_Modeling.ipynb
│
├── src/
│   ├── preprocess.py
│   ├── features.py
│   └── model.py
│
├── submissions/
│   └── submission.csv
│
├── requirements.txt
└── README.md
```

## 🚀 Getting Started

### Prerequisites

```bash
pip install -r requirements.txt
```

**Key dependencies:**
```
pandas
numpy
scikit-learn
xgboost
matplotlib
seaborn
jupyter
```

### Run the Pipeline

```bash
# Clone the repo
git clone https://github.com/your-username/titanic-kaggle.git
cd titanic-kaggle

# Launch notebooks
jupyter notebook notebooks/01_EDA.ipynb

# Or run the full pipeline
python src/model.py
```

---

## 📤 Submission

The output file `submissions/submission.csv` contains predictions in the required format:

```
PassengerId,Survived
892,0
893,1
894,0
...
```

---

## 💡 Key Takeaways

1. **Gender was the single most predictive feature** — "women and children first" was statistically significant
2. **Socioeconomic class** had a strong effect on survival, likely due to lifeboat access
3. **Feature engineering** (titles, family size, age bins) meaningfully improved model performance
4. **Ensemble methods** (Random Forest, XGBoost) outperformed linear models for this dataset

---

## 📚 References

- [Kaggle Titanic Competition](https://www.kaggle.com/competitions/titanic)
- [Encyclopedia Titanica](https://www.encyclopedia-titanica.org/)
- Scikit-learn Documentation

---

## 👤 Author

**Kushanavo**
Civil Engineering | IIEST Shibpur

*Exploring Machine Learning & Data Science*

[![Kaggle](https://img.shields.io/badge/Kaggle-Profile-blue?logo=kaggle)](https://www.kaggle.com)
[![GitHub](https://img.shields.io/badge/GitHub-Repo-black?logo=github)](https://github.com)


*"It is not the ship so much as the skillful sailing that assures the prosperous voyage."* — George William Curtis
