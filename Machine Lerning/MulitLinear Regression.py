import numpy as np
import pandas as pd
import statsmodels.formula.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

df = pd.read_csv('startups.csv')

X = df[df.columns[:-1]]
y = df['Profit']

le = LabelEncoder()

X['State'] = le.fit_transform(X['State'])

ohe = OneHotEncoder()
state_array = ohe.fit_transform(X["State"].values.reshape(-1, 1)).toarray()[:, :-1]
states = pd.DataFrame(state_array, columns=le.classes_[:-1])
X = pd.concat([states, X], axis=1)
X.drop('State', inplace=True, axis=1)
print(X.head())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.1, random_state=101)

regressor = LinearRegression()
regressor.fit(X_train, y_train)

pred = regressor.predict(X_test)

print(pred)
print(y_test)
