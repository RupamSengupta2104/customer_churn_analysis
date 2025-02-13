import pandas as pd
from pathlib import Path  

# Define base directory & file path
base_dir = Path(__file__).resolve().parent.parent  
file_path = base_dir / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"  

# Load dataset again
df = pd.read_csv(file_path)

# Replace empty spaces with NaN (TotalCharges had empty strings)
df = df.replace(" ", pd.NA)

# Convert TotalCharges to numeric (Fix Empty Strings)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)

# Drop customerID (it’s not useful for ML)
df.drop(columns=["customerID"], inplace=True)

# Convert 'Churn' to binary (Yes = 1, No = 0)
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# List of categorical columns
categorical_columns = ["Partner", "Dependents", "PhoneService", "MultipleLines", 
                       "InternetService", "OnlineSecurity", "OnlineBackup", 
                       "DeviceProtection", "TechSupport", "StreamingTV", 
                       "StreamingMovies", "Contract", "PaperlessBilling", 
                       "PaymentMethod"]

# Convert categorical values to lowercase & strip spaces
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.str.strip().str.lower())

# One-hot encoding for categorical columns (Drop first category to avoid multicollinearity)
df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

# Save cleaned dataset
cleaned_file_path = base_dir / "data" / "cleaned_customer_churn.csv"
df.to_csv(cleaned_file_path, index=False)

print("✅ Cleaned dataset saved successfully!")
