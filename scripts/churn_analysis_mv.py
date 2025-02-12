from pathlib import Path  # Used to handle file paths
import pandas as pd  # Used for data analysis

# Get the absolute path of the project directory
base_dir = Path(__file__).resolve().parent.parent  # Moves up two levels
file_path = base_dir / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"  # Path to data file

# Load dataset into a pandas DataFrame
df = pd.read_csv(file_path)
'''
# Display first 5 rows
print(df.head())

# Show dataset structure & column data types
print(df.info())
'''
# Count missing values per column
missing_values = df.isnull().sum()

# Display only columns that have missing values
print(missing_values[missing_values > 0])
'''
# Check for empty strings
print((df == "").sum())

# Replace empty spaces with NaN (since some missing values are blank spaces)
df = df.replace(" ", pd.NA)

# Check for Special Missing Indicators
print(df.isin(["NA", "N/A", "?", "-"]).sum())

# Check for Spaces or Special Characters in Object Columns
print((df.applymap(lambda x: x.strip() if isinstance(x, str) else x) == "").sum())

print(df["TotalCharges"].dtype)

# Convert TotalCharges to Numeric (Fix Empty Strings)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Fill with 0 (if missing means unpaid)
df["TotalCharges"] = df["TotalCharges"].fillna(0)

print(df.isnull().sum())


# Check unique values in categorical columns
categorical_columns = ["Partner", "Dependents", "PhoneService", "MultipleLines", 
                       "InternetService", "OnlineSecurity", "OnlineBackup", 
                       "DeviceProtection", "TechSupport", "StreamingTV", 
                       "StreamingMovies", "Contract", "PaperlessBilling", 
                       "PaymentMethod", "Churn"]

for col in categorical_columns:
    print(f"Unique values in {col}: {df[col].unique()}")

# If inconsistencies (like Yes, yes, No, no), convert them to lowercase and standardize them
# Standardize categorical values
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.str.strip().str.lower())
# Now, all categorical values are consistent (yes/no instead of variations)
'''
## Convert Categorical Variables

# Convert 'Churn' column to binary (Yes = 1, No = 0)
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# One-hot encoding for categorical columns
df = pd.get_dummies(df, drop_first=True)

# Show first few rows after transformation
print(df.head())

# Save the cleaned dataset
cleaned_file_path = base_dir / "data" / "cleaned_customer_churn.csv"
df.to_csv(cleaned_file_path, index=False)

print("Cleaned dataset saved successfully!")
