import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Read the dataset
data = pd.read_csv('Part_E\data\solar_weather.csv')


# Display the summary statistics of the data
print(data.describe())

# Plot distribution of energy Production
plt.figure(figsize=(10, 6))
sns.histplot(data['Energy delta[Wh]'], bins=20, kde=True)
plt.title('Distribution of Energy Production')
plt.xlabel('Energy Production (Wh)')
plt.ylabel('Frequency')
plt.show()

# Explore relationship between energy Production and weather parameters
# Scatter plot between energy Production and temperature
plt.figure(figsize=(10, 6))
sns.scatterplot(x='GHI', y='Energy delta[Wh]', data=data)
plt.title('Energy Production vs Solar radiation (GHI)')
plt.xlabel('Solar Radiation (GHI)')
plt.ylabel('Energy Production (Wh)')
plt.show()

# Boxplot of energy Production by weather type
plt.figure(figsize=(10, 6))
sns.boxplot(x='weather_type', y='Energy delta[Wh]', data=data)
plt.title('Energy Production by Weather Type')
plt.xlabel('Weather Type')
plt.ylabel('Energy Production (Wh)')
plt.xticks(rotation=45)
plt.show()

# Exclude the first column from the DataFrame
data_without_first_column = data.iloc[:, 1:]

# Correlation Matrix plot
plt.figure(figsize=(10, 6))
sns.heatmap(data_without_first_column.corr(), annot=True, fmt=".2f", cmap='coolwarm')
plt.title("Correlation Matrix of Features (Excluding First Column)")
plt.show()


