import pandas as pd
from pathlib import Path

# Define base directory
base_dir = Path(__file__).resolve().parent  # Gets the script's directory
file_path = base_dir.parent / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

# Check if file exists before loading
if not file_path.exists():
    raise FileNotFoundError(f"Dataset not found at: {file_path}")

# Load dataset
df = pd.read_csv(file_path)

# Display first 5 rows
print("\n🔹 First 5 Rows of Dataset:")
print(df.head())

# Dataset information
print("\n🔹 Dataset Info:")
df.info()

# Count missing values in each column
print("\n🔹 Missing Values Count:")
print(df.isnull().sum())

# Summary statistics
print("\n🔹 Summary Statistics:")
print(df.describe())

# Knowing data types
print("\n🔹 Column Data Types:")
print(df.dtypes)

# To see only numeric columns (int & float)
print("\n🔹 Numeric Columns Sample:")
print(df.select_dtypes(include=['int64', 'float64']).head())
