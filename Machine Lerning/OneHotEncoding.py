import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder

# reading the dataset for OneHotEncoding Example
df = pd.read_csv('oneHotEnc.csv')
print(df.head())

nan = df[['Age', 'Salary']]


# processing missing value
print("Processing Missing Values with sklearn Imputer")
impute = Imputer(missing_values='NaN')
impute.fit(nan)
X = impute.transform(nan)
df[['Age', 'Salary']] = X
print(df)

# Label Encoding
print("Label Encoding... ")
country = df['Country'].values
label_enc = LabelEncoder()
label_enc.fit(country)
country = label_enc.transform(country)
df['Country'] = country
print(df)

# One Hot Encoding
print("One Hot Encoding...")
X = df['Country'].values
X = label_enc.fit_transform(X)
Xenc = OneHotEncoder()
Xenc.fit(X.reshape(-1, 1))
X = Xenc.transform(X.reshape(-1, 1)).toarray()
print(X)
