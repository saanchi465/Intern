import pandas as pd
import matplotlib.pyplot as plt
import os

csv_data = """Product,Sales,Advertising,Profit
A,200,50,40
B,150,40,30
C,300,70,60
D,250,65,55
E,180,45,35
"""

with open("data.csv", "w") as file:
    file.write(csv_data)

print("CSV file created successfully!")
print("Files in folder:", os.listdir())

data = pd.read_csv("data.csv")

print("\nFirst 5 rows of dataset:")
print(data.head())

average_sales = data['Sales'].mean()
print("\nAverage Sales:", average_sales)

plt.figure()
plt.bar(data['Product'], data['Sales'])
plt.xlabel("Product")
plt.ylabel("Sales")
plt.title("Sales by Product")
plt.show()

plt.figure()
plt.scatter(data['Advertising'], data['Sales'])
plt.xlabel("Advertising Cost")
plt.ylabel("Sales")
plt.title("Advertising vs Sales")
plt.show()

correlation = data[['Sales', 'Advertising', 'Profit']].corr()

plt.figure()
plt.imshow(correlation)
plt.colorbar()
plt.xticks(range(len(correlation.columns)), correlation.columns)
plt.yticks(range(len(correlation.columns)), correlation.columns)
plt.title("Correlation Heatmap")
plt.show()
