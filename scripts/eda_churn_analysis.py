import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Define directories
base_dir = Path.cwd()
data_dir = base_dir / "data"
images_dir = base_dir / "images"

# Create images directory if it doesn't exist
images_dir.mkdir(parents=True, exist_ok=True)

# Load the cleaned dataset
file_path = data_dir / "cleaned_customer_churn.csv"
df = pd.read_csv(file_path)

# Standardize column names (strip spaces)
df.columns = df.columns.str.strip()

# Dataset Overview
print(df.info())
print(df.head())

# Ensure Data Types Are Correct
if "SeniorCitizen" in df.columns and df["SeniorCitizen"].dtype != "object":
    df["SeniorCitizen"] = df["SeniorCitizen"].astype(str)

# Extract categorical columns dynamically
categorical_columns = df.select_dtypes(include=["object"]).columns.tolist()

if categorical_columns:
    print("\nCategorical Features Summary:")
    print(df.describe(include="object"))
else:
    print("No categorical columns found in the dataset.")

# Extract numerical columns
numerical_columns = df.select_dtypes(include=["number"]).columns.tolist()

# Plot and save histograms for numerical features
if numerical_columns:
    df[numerical_columns].hist(figsize=(12, 8), bins=20, edgecolor="black")
    plt.suptitle("Distribution of Numerical Features", fontsize=14)
    plt.savefig(images_dir / "numerical_distributions.png")
    plt.show()

# Bar plots for categorical features (fixing Seaborn warning)
if categorical_columns:
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(categorical_columns, 1):
        plt.subplot(5, 4, i)
        sns.countplot(data=df, x=col, hue=col, palette="viridis", legend=False)
        plt.xticks(rotation=45)
        plt.title(f"Distribution of {col}")
    plt.tight_layout()
    plt.savefig(images_dir / "categorical_distributions.png")
    plt.show()

# Correlation Analysis (only for numeric features)
if numerical_columns:
    corr_matrix = df[numerical_columns].corr()
    plt.figure(figsize=(12, 6))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
    plt.title("Feature Correlation Matrix")
    plt.savefig(images_dir / "correlation_matrix.png")
    plt.show()

# Churn vs Numerical Features
if "Churn" in df.columns and numerical_columns:
    plt.figure(figsize=(12, 4))
    for i, col in enumerate(numerical_columns, 1):
        plt.subplot(1, len(numerical_columns), i)
        sns.boxplot(data=df, x="Churn", y=col, palette="coolwarm")
        plt.title(f"{col} vs Churn")
    plt.tight_layout()
    plt.savefig(images_dir / "churn_vs_numerical.png")
    plt.show()

# Churn vs Categorical Features (only if "Churn" exists)
if "Churn" in df.columns and categorical_columns:
    plt.figure(figsize=(15, 8))
    for i, col in enumerate(categorical_columns[:3], 1):  # Select 3 categorical columns for comparison
        plt.subplot(1, 3, i)
        sns.countplot(data=df, x=col, hue="Churn", palette="coolwarm")
        plt.xticks(rotation=45)
        plt.title(f"Churn Distribution by {col}")
    plt.tight_layout()
    plt.savefig(images_dir / "churn_vs_categorical.png")
    plt.show()

print("Exploratory Data Analysis (EDA) completed! All plots saved in the 'images' folder.")

