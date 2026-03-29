import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression

# Load dataset
weather = pd.read_csv("weather_data.csv")

print(weather.head())

# Pearson Correlation (Temperature vs Humidity)
corr, p = pearsonr(weather["MaxTemp"], weather["Humidity"])
print("Pearson Correlation:", corr)

# Spearman Correlation (Rainfall vs WindSpeed)
corr2, p2 = spearmanr(weather["Rainfall"], weather["WindSpeed"])
print("Spearman Correlation:", corr2)

# Simple Linear Regression
X = weather[["Humidity"]]
y = weather["MaxTemp"]

model = LinearRegression()
model.fit(X, y)

print("Intercept:", model.intercept_)
print("Coefficient:", model.coef_[0])

# Multiple Regression
X2 = weather[["Humidity","WindSpeed"]]
model2 = LinearRegression()
model2.fit(X2, y)

print("Multiple Regression Coefficients:", model2.coef_)

# R squared
print("R squared:", model2.score(X2, y))

# Scatter plot
plt.scatter(weather["Humidity"], weather["MaxTemp"])

plt.xlabel("Humidity")
plt.ylabel("Max Temperature")
plt.title("Temperature vs Humidity")

m = model.coef_[0]
c = model.intercept_

plt.plot(weather["Humidity"], m*weather["Humidity"] + c)

plt.show()

# Correlation heatmap
corr_matrix = weather.corr(numeric_only=True)

sns.heatmap(corr_matrix, annot=True)

plt.title("Weather Correlation Matrix")
plt.show()