import pandas as pd

# Read the dataset
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Check the first few rows of the dataset
print(df.head())

# Check the summary statistics of the numerical columns
print(df.describe())

# Check the data types of each column
print(df.dtypes)

# Handle missing values
print(df.isnull().sum())

# Replace missing values in 'TotalCharges' column with 0
df['TotalCharges'].fillna(0, inplace=True)

# Perform feature engineering
# For example, create a new binary column 'HasInternetService' based on 'InternetService' column
df['HasInternetService'] = df['InternetService'].apply(lambda x: 1 if x != 'No' else 0)

# Conduct exploratory data analysis
# Calculate churn rate
churn_rate = df['Churn'].value_counts() / len(df) * 100
print("Churn Rate:\n", churn_rate)
