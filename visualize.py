# Visualize churn distribution
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
df['Churn'].value_counts().plot(kind='bar')
plt.xlabel('Churn')
plt.ylabel('Count')
plt.title('Churn Distribution')
plt.show()

# Analyze the relationship between churn and other variables
# For example, calculate the average monthly charges for churned and non-churned customers
avg_monthly_charges = df.groupby('Churn')['MonthlyCharges'].mean()
print("Average Monthly Charges:\n", avg_monthly_charges)
