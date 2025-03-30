# 💧 Water Pump Predictor / Predicción del Estado de Bombas de Agua

**DrivenData Challenge: "Pump it Up - Data Mining the Water Table"**  
Score obtenido en la leaderboard pública: **0.8185**

---

##  Project Overview / Descripción del Proyecto

**ENGLISH:**  
This machine learning project aims to predict the functional status of water pumps in rural Tanzania using structured data. It is based on the DrivenData competition and uses models like Random Forest, XGBoost and LightGBM with tailored preprocessing and feature engineering.

**ESPAÑOL:**  
Este proyecto de machine learning tiene como objetivo predecir el estado funcional de bombas de agua en zonas rurales de Tanzania. Se basa en la competición de DrivenData y utiliza modelos como Random Forest, XGBoost y LightGBM, aplicando técnicas de preprocesado y optimización de hiperparámetros.

---

##  Final Score / Resultado Final

- ✅ Model: **Random Forest Classifier**
- ✅ Public Score: **0.8185**
- ✅ Feature engineering: population log, construction year, amount_tsh, etc.
- ✅ Handling of missing values with **KNNImputer**
- ✅ Encoding with **TargetEncoder**
- ✅ Hyperparameter tuning with **RandomizedSearchCV**

---

## 📂 Repository Structure / Estructura del Repositorio

```
├── data/                     # Archivos CSV (train, test, labels)
├── outputs/                  # submission_8185.csv generado
├── visuals/                  # Gráficos generados (importancia, confusión, EDA)
├── scripts/                  # Scripts de modelo, validación y comparación
│   ├── model_final_en.py
│   ├── 01_EDA.py
│   ├── 02_Model_Validation.py
│   └── 03_Model_Comparison_LGBM.py
├── config.yaml               # Parámetros de configuración
├── requirements.txt          # Dependencias
├── README.md                 # Este archivo
├── CHANGELOG.md              # Historial de versiones
├── CONTRIBUTING.md           # Guía para contribuciones
└── LICENSE
```

---

##  How to Run / Cómo Ejecutar

1. ✅ Clone the repo / Clona el repositorio:
```bash
git clone https://github.com/Haiubabdelkader99/water-pump-predictor.git
cd water-pump-predictor
```

2. ✅ Install dependencies / Instala las dependencias:
```bash
pip install -r requirements.txt
```

3. ✅ Run any script / Ejecuta cualquier script:
```bash
python scripts/model_final_en.py
```

---

## 🌐 Streamlit App (Coming Soon) / Aplicación Interactiva

**ENGLISH:** A Streamlit interactive app will allow CSV upload or manual input to predict pump status in real-time.

**ESPAÑOL:** Próximamente se añadirá una app interactiva en Streamlit para cargar archivos CSV o introducir valores manuales y predecir el estado de la bomba en tiempo real.

---

## 📧 Contact

Developed by **Haiub Abdelkader**  
🔗 [LinkedIn](https://linkedin.com/in/haiubabdelkader)
