"""
Model Comparison with Hyperparameter Tuning

This script compares the performance of several classification models:
- Random Forest
- Logistic Regression
- XGBoost
- LightGBM

It also demonstrates how hyperparameter tuning using RandomizedSearchCV can help improve model performance.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer
from category_encoders import TargetEncoder

sns.set_style("whitegrid")

# Load and preprocess data
path = "C:/Users/Haiub/Desktop/water-pump-predictor/"
X = pd.read_csv(path + "Trainingsetvalues.csv")
y = pd.read_csv(path + "Trainginsetlabels.csv")
df = pd.merge(X, y, on="id")
label_encoder = LabelEncoder()
df["status_group_encoded"] = label_encoder.fit_transform(df["status_group"])
X = df.drop(columns=["status_group", "status_group_encoded", "id"])
y = df["status_group_encoded"]

# Handle missing values and encode categories
num_cols = X.select_dtypes(include=["int64", "float64"]).columns
cat_cols = X.select_dtypes(include=["object", "bool"]).columns
X[num_cols] = KNNImputer(n_neighbors=5).fit_transform(X[num_cols])
X[cat_cols] = TargetEncoder().fit_transform(X[cat_cols], y)

# Train/Test split
X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Initialize models
models = {
    "RandomForest": RandomForestClassifier(n_estimators=200, max_depth=25, class_weight="balanced", random_state=42),
    "LogisticRegression": LogisticRegression(max_iter=1000, multi_class="multinomial", solver="lbfgs"),
    "XGBoost": XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, objective="multi:softmax",
                             num_class=3, eval_metric="mlogloss", verbosity=0),
    "LightGBM": LGBMClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, class_weight="balanced", random_state=42)
}

# Store evaluation results
results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average='weighted')
    results.append({"Model": name, "Accuracy": acc, "F1 Score": f1})
    print(f"{name} - Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")

# Optional: Hyperparameter tuning for LightGBM
# Why hyperparameter tuning? It helps find the best model settings that improve performance and reduce overfitting.
param_dist = {
    "n_estimators": [100, 200, 300],
    "max_depth": [6, 10, 15],
    "learning_rate": [0.01, 0.05, 0.1],
    "num_leaves": [31, 50, 100]
}

tuner = RandomizedSearchCV(
    estimator=LGBMClassifier(class_weight="balanced", random_state=42),
    param_distributions=param_dist,
    n_iter=10,
    cv=3,
    scoring="f1_weighted",
    random_state=42,
    n_jobs=-1
)

tuner.fit(X_train, y_train)
best_lgbm = tuner.best_estimator_
y_pred_tuned = best_lgbm.predict(X_val)
acc = accuracy_score(y_val, y_pred_tuned)
f1 = f1_score(y_val, y_pred_tuned, average='weighted')
results.append({"Model": "LightGBM Tuned", "Accuracy": acc, "F1 Score": f1})
print(f"LightGBM Tuned - Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")
print("Best Params:", tuner.best_params_)

# Plot comparison
results_df = pd.DataFrame(results)
sns.barplot(data=results_df.melt(id_vars="Model"), x="Model", y="value", hue="variable")
plt.title("Model Performance Comparison (with Tuning)")
plt.ylabel("Score")
plt.ylim(0.6, 1.0)
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("visuals/model_comparison_lgbm_tuned.png")
plt.show()
