import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Reading the dataset
df = pd.read_csv('Salary_Data.csv')
print(df.info())

# making Dependent and Independent variable
X = df['YearsExperience'].values
y = df['Salary'].values

# splitting the dataset into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)

# fitting data into Linear Regression
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train.reshape(-1, 1), y_train.reshape(-1, 1))

# predicting the results
pred = regressor.predict(X_test.reshape(-1, 1))

# visualizing the prediction result
# plt.scatter(X_train.reshape(-1, 1), y_train.reshape(-1, 1), c='red')
plt.scatter(X.reshape(-1, 1), y.reshape(-1, 1), c='red')
plt.plot(X_train.reshape(-1, 1), regressor.predict(X_train.reshape(-1, 1)))
plt.title("Salary vs Experience (Training set)")
plt.xlabel("Experience(years)")
plt.ylabel("Salary")
plt.show()
