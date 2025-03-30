import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, classification_report, ConfusionMatrixDisplay
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer
from category_encoders import TargetEncoder

# Set style
sns.set_style("whitegrid")

# Paths
path = "C:/Users/Haiub/Desktop/water-pump-predictor/"
X = pd.read_csv(path + "Trainingsetvalues.csv")
y = pd.read_csv(path + "Trainginsetlabels.csv")

# Merge
df = pd.merge(X, y, on="id")
label_encoder = LabelEncoder()
df["status_group_encoded"] = label_encoder.fit_transform(df["status_group"])

# Split features and target
X = df.drop(columns=["status_group", "status_group_encoded", "id"])
y = df["status_group_encoded"]

# Separate types
num_cols = X.select_dtypes(include=["int64", "float64"]).columns
cat_cols = X.select_dtypes(include=["object", "bool"]).columns

# Impute numerical
X[num_cols] = KNNImputer(n_neighbors=5).fit_transform(X[num_cols])

# Encode categoricals
encoder = TargetEncoder()
X[cat_cols] = encoder.fit_transform(X[cat_cols], y)

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=200, max_depth=25, random_state=42,
                               class_weight="balanced", n_jobs=-1)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_val)

# Metrics
print("📋 Classification Report:")
print(classification_report(y_val, y_pred, target_names=label_encoder.classes_))

# Confusion matrix
cm = confusion_matrix(y_val, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("visuals/confusion_matrix.png")
plt.show()
