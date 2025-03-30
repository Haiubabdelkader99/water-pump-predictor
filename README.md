# 🚰 Pump It Up: Predicting Water Pump Status in Tanzania

This repository contains a machine learning project developed for the DrivenData competition: **"Pump It Up: Data Mining the Water Table"**.

The goal is to predict the functional status of water pumps in Tanzania based on technical, geographic, and demographic features.

## 📈 Final Score

✅ Public Leaderboard Score: **0.8185**

## 📂 Project Structure

```
pump-it-up-ml/
├── data/                    # Data location instructions (no raw files shared)
├── scripts/                 # Final training scripts
├── outputs/                 # Final CSV submission
├── visuals/                 # Feature importance plots and graphs
├── README.md                # Project documentation
├── requirements.txt         # Required Python packages
└── memo_modelo_8185.pdf     # Professional memo of the final model
```

## 💻 Technologies Used

- Python 3.13
- pandas, NumPy
- scikit-learn
- matplotlib

## 🔧 Data Preprocessing & Feature Engineering

- Removed low-impact columns: `recorded_by`, `wpt_name`, `scheme_name`, `num_private`
- Handled missing values (mean for numerical, 'missing' for categorical)
- Created new features:
  - `years_old`: age of the pump
  - `tsh_per_capita`: investment per person
  - `log_population`: to reduce outlier impact
  - `has_funder`: binary indicator
- Used `OrdinalEncoder` for categorical variables

## 🤖 Model Description

- Model: `RandomForestClassifier`
  - `n_estimators = 300`
  - `max_depth = 25`
  - `class_weight = 'balanced'`

## 📊 Feature Importance

<img src="visuals/feature_importance.png" width="600">

## 🔁 Next Steps

- Apply GridSearchCV for hyperparameter tuning
- Test more advanced models: XGBoost, LightGBM
- Try ensemble methods and feature selection

## 🧪 How to Reproduce

1. Clone the repository
2. Place the dataset CSVs in the `data/` folder
3. Run the script in `scripts/model_final.py`
4. Submission file will be saved in `outputs/submission_8185.csv`

## 👤 Author

Haiub  
March 30, 2025
