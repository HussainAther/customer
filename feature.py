import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

## Feature engineering 

# Read the dataset
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Drop unnecessary columns (if any)
df = df.drop('customerID', axis=1)

# Convert categorical variables to numerical using one-hot encoding
df = pd.get_dummies(df, drop_first=True)

# Separate the features (X) and target variable (y)
X = df.drop('Churn_Yes', axis=1)  # Features
y = df['Churn_Yes']  # Target variable

# Feature extraction using SelectKBest and chi-square test
best_features = SelectKBest(score_func=chi2, k=10)
fit = best_features.fit(X, y)
feature_scores = pd.DataFrame({'Feature': X.columns, 'Score': fit.scores_}).sort_values(by='Score', ascending=False)

# Plot feature importance scores
plt.figure(figsize=(10, 6))
sns.barplot(x='Score', y='Feature', data=feature_scores)
plt.title('Feature Importance Scores')
plt.xlabel('Score')
plt.ylabel('Feature')
plt.show()

# Select the top k features
k = 5  # Number of top features to select
selected_features = feature_scores['Feature'][:k].tolist()
print("Selected Features:\n", selected_features)

