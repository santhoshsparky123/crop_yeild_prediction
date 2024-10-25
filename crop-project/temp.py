import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('/content/crop_yield.csv')

# One-hot encoding categorical variables
df_encoded = pd.get_dummies(df, columns=['Crop', 'Season', 'State'])

# Correlation matrix
plt.figure(figsize=(10, 6))
sns.heatmap(df_encoded.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()

# Plotting production over the years
plt.figure(figsize=(8,5))
sns.lineplot(x='Crop_Year', y='Production', data=df)
plt.title('Production over the Years')
plt.show()

# Features (X) and target (y)
X = df_encoded.drop(['Production'], axis=1)
y = df_encoded['Production']

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate Mean Squared Error and R-squared
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# Comparison of actual vs predicted values
comparison_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(comparison_df.head())

# Plot actual vs predicted values
plt.figure(figsize=(10,6))
plt.plot(y_test.values, label='Actual', color='b', marker='o')  # Actual values
plt.plot(y_pred, label='Predicted', color='r', marker='x')  # Predicted values
plt.title('Actual vs Predicted Production Values')
plt.xlabel('Sample index')
plt.ylabel('Production')
plt.legend()
plt.show()

# Prepare future data for prediction
future_data = pd.DataFrame({
    'Crop_jute': [1],  # One-hot encoded crop value (adjust based on your dataset)
    'Crop_Year': [2000],
    'Season_Rabi': [1],  # One-hot encoded season value
    'State_Assam': [1],  # One-hot encoded state value
    'Area': [19880],
    'Annual_Rainfall': [2015],
    'Fertilizer': [7188990],
    'Pesticide': [4332],
    'Yield': [2.3456]
})

# Align future data with the training data (adding missing columns if necessary)
future_data = future_data.reindex(columns=X_train.columns, fill_value=0)

# Predict future production
future_prediction = model.predict(future_data)
print(f'Predicted Production for 2000: {future_prediction[0]}')
