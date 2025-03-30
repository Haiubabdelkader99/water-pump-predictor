import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer
from category_encoders import TargetEncoder

# 1. Cargar datos
ruta = "C:/Users/Haiub/Desktop/ML/"
X_train = pd.read_csv(ruta + "Trainingsetvalues.csv")
y_train = pd.read_csv(ruta + "Trainginsetlabels.csv")
X_test = pd.read_csv(ruta + "Testsetvalues.csv")

# 2. Unir X e y
df = pd.merge(X_train, y_train, on="id")
label_encoder = LabelEncoder()
df["status_group_encoded"] = label_encoder.fit_transform(df["status_group"])

# 3. Separar features/target
X = df.drop(columns=["status_group", "status_group_encoded"])
y = df["status_group_encoded"]

# 4. Concatenar con test para tratar igual
X_all = pd.concat([X, X_test], axis=0)
X_all = X_all.drop(columns=["id"])  # Quitamos id, lo recuperamos luego

# 5. Separar numéricas y categóricas
num_cols = X_all.select_dtypes(include=["int64", "float64"]).columns
cat_cols = X_all.select_dtypes(include=["object", "bool"]).columns

# 6. Imputar valores numéricos con KNN
X_all[num_cols] = KNNImputer(n_neighbors=5).fit_transform(X_all[num_cols])

# 7. Codificar categóricas con TargetEncoder solo en el train
X_cat = X_all[cat_cols].astype(str)
encoder = TargetEncoder()

X_cat_encoded = encoder.fit_transform(
    X_cat.iloc[:len(X)], y
)
X_cat_test_encoded = encoder.transform(X_cat.iloc[len(X):])

# 8. Concatenar todo de nuevo
X_train_final = pd.concat([pd.DataFrame(X_all[num_cols].iloc[:len(X)].values, columns=num_cols),
                           X_cat_encoded.reset_index(drop=True)], axis=1)

X_test_final = pd.concat([pd.DataFrame(X_all[num_cols].iloc[len(X):].values, columns=num_cols),
                          X_cat_test_encoded.reset_index(drop=True)], axis=1)

# 9. Entrenar modelo con class_weight balanced
model = RandomForestClassifier(n_estimators=200, max_depth=25,
                               random_state=42, class_weight='balanced', n_jobs=-1)
model.fit(X_train_final, y)

# 10. Predecir test y generar submission
y_pred_test = model.predict(X_test_final)
submission = pd.DataFrame({
    "id": X_test["id"],
    "status_group": label_encoder.inverse_transform(y_pred_test)
})
submission.to_csv(ruta + "submission_mejorado.csv", index=False)
print("✅ Submission mejorada generada: submission_mejorado.csv")
