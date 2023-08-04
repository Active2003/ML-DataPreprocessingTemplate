# Data Preprocessing

# Importing the libraries
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset from 'Data.csv' file
dataset = pd.read_csv('Data.csv')

# Extracting all the independent columns (features) into variable X
# iloc[:, :-1]: Select all rows and all columns except the last one
X = dataset.iloc[:, :-1].values

# Extracting the dependent column (target) into variable Y
# iloc[:, 3]: Select all rows and the 4th column (index 3)
Y = dataset.iloc[:, 3].values

# Taking care of missing data using SimpleImputer from scikit-learn

# Creating an instance of SimpleImputer with missing values as NaN and strategy as 'mean'
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

# Fitting the imputer on the relevant columns (2nd and 3rd columns) of X
imputer = imputer.fit(X[:, 1:3])

# Replacing the missing data in the 2nd and 3rd columns of X with the mean value of the respective columns
X[:, 1:3] = imputer.transform(X[:, 1:3])

# Encoding categorical data

# Creating the LabelEncoder object for the categorical feature (1st column)
labelencoder_X = LabelEncoder()

# Fit labelencoder_X object to the country column (1st column) and transform the country column
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])

# Creating the ColumnTransformer for one-hot encoding
# This will one-hot encode the 1st column (index 0) using OneHotEncoder,
# while keeping the other columns unchanged with 'passthrough'
ct = ColumnTransformer([("Country", OneHotEncoder(), [0])], remainder='passthrough')

# Fit and transform the data using the ColumnTransformer
X = ct.fit_transform(X)

# Creating the LabelEncoder object for the dependent column (target)
labelencoder_Y = LabelEncoder()

# Fit labelencoder_Y object to the dependent column (target) and transform it
Y = labelencoder_Y.fit_transform(Y)

# No need to convert the result to a dense array since it is already handled by ColumnTransformer

# splitting dataset into the traning and test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

# Feature scaling
from sklearn.preprocessing import StandardScaler
#sc_X is object
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
