import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set a consistent style
sns.set_style("whitegrid")
sns.set_palette("deep")

# Define data path
data_path = "C:/Users/Haiub/Desktop/water-pump-predictor/"
train_values_file = os.path.join(data_path, 'Trainingsetvalues.csv')
train_labels_file = os.path.join(data_path, 'Trainginsetlabels.csv')

# Load data
train_values = pd.read_csv(train_values_file)
train_labels = pd.read_csv(train_labels_file)
df = pd.merge(train_values, train_labels, on='id')

# Create visuals folder
os.makedirs("visuals", exist_ok=True)

# Print basic info
print(f"Shape: {df.shape}")
print("Target distribution:\n", df['status_group'].value_counts())

# 1. Histograms
numeric_cols = df.select_dtypes(include='number').columns.tolist()
if 'id' in numeric_cols:
    numeric_cols.remove('id')

for col in numeric_cols:
    plt.figure(figsize=(8, 6))
    sns.histplot(data=df, x=col, bins=30, color='blue')
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(f"visuals/hist_{col}.png")
    plt.show()
    plt.close()

# 2. Scatter plot longitude vs latitude
plt.figure(figsize=(8, 6))
scatter_data = df[df['longitude'] != 0]
sns.scatterplot(x='longitude', y='latitude', data=scatter_data, alpha=0.3, marker='o', edgecolor='none')
plt.title("Longitude vs Latitude")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.tight_layout()
plt.savefig("visuals/scatter_longitude_latitude.png")
plt.show()
plt.close()

# 3. Boxplots by status_group
key_cols = ['gps_height', 'population', 'construction_year']
for col in key_cols:
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='status_group', y=col, data=df)
    plt.title(f"{col} by status_group")
    plt.xlabel("status_group")
    plt.ylabel(col)
    plt.tight_layout()
    plt.savefig(f"visuals/boxplot_{col}_by_status.png")
    plt.show()
    plt.close()

# 4. Missing values
missing_counts = df.isnull().sum()
missing_counts = missing_counts[missing_counts > 0].sort_values(ascending=False)
if not missing_counts.empty:
    plt.figure(figsize=(8, 5))
    sns.barplot(x=missing_counts.values, y=missing_counts.index, color='blue')
    plt.title("Missing Values per Feature")
    plt.xlabel("Missing Count")
    plt.ylabel("Feature")
    for index, value in enumerate(missing_counts.values):
        plt.text(value + 100, index, str(value), va='center', fontsize=9)
    plt.tight_layout()
    plt.savefig("visuals/missing_values.png")
    plt.show()
    plt.close()
else:
    print("No missing values.")

# 5. Correlation matrix
numeric_df = df.select_dtypes(include=["int64", "float64"])
plt.figure(figsize=(10, 8))
sns.heatmap(numeric_df.corr(), annot=False, cmap="coolwarm", fmt=".2f", square=True)
plt.title("Correlation Matrix (Numeric Features)")
plt.tight_layout()
plt.savefig("visuals/numeric_correlation_matrix.png")
plt.show()
plt.close()
