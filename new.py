import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os

csv_data = """OverallQual,GrLivArea,BedroomAbvGr,SalePrice
7,1710,3,208500
6,1262,3,181500
7,1786,3,223500
7,1717,3,140000
8,2198,4,250000
5,1362,2,143000
8,1694,3,307000
7,2090,3,200000
6,1774,3,129900
5,1077,2,118000
"""

with open("house_prices.csv", "w") as file:
    file.write(csv_data)

print("Dataset created successfully!")
print("Files in folder:", os.listdir())

data = pd.read_csv("house_prices.csv")
print("\nDataset Preview:")
print(data.head())

X = data[['OverallQual', 'GrLivArea', 'BedroomAbvGr']]
y = data['SalePrice']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("\nMean Absolute Error:", mean_absolute_error(y_test, predictions))
print("Mean Squared Error:", mean_squared_error(y_test, predictions))

plt.figure()
plt.scatter(y_test, predictions)
plt.xlabel("Actual House Price")
plt.ylabel("Predicted House Price")
plt.title("Actual vs Predicted House Prices")
plt.show()
