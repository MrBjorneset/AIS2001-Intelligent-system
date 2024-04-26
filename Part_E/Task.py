import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('data/solar_weather.csv')

# Display first few rows 
print(data.head())


# Display the data types of each column
print(data.info())

# Display the summary statistics of the data
print(data.describe())

# Data Visualization
# Plot distribution of energy consumption
plt.figure(figsize=(10, 6))
sns.histplot(data['Energy delta[Wh]'], bins=20, kde=True)
plt.title('Distribution of Energy Consumption')
plt.xlabel('Energy Consumption (Wh)')
plt.ylabel('Frequency')
plt.show()

# Explore relationship between energy consumption and weather parameters
# For example, scatter plot between energy consumption and temperature
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Temperature', y='Energy delta[Wh]', data=data)
plt.title('Energy Consumption vs Temperature')
plt.xlabel('Temperature (°C)')
plt.ylabel('Energy Consumption (Wh)')
plt.show()

# Visualize distribution of weather types
plt.figure(figsize=(10, 6))
sns.countplot(x='weather_type', data=data, palette='viridis')
plt.title('Distribution of Weather Types')
plt.xlabel('Weather Type')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# Further Analysis
# Calculate correlations between energy consumption and weather parameters
correlation_matrix = data.corr()
print(correlation_matrix)
