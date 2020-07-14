# Data Preprocessing Template

# Importing the libraries
import numpy as np # allows to work with arrays
import matplotlib.pyplot as plt # allows us to plot charts
import pandas as pd # Allows us to import datasets and independant variable? - need more information 

# Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values #feature
y = dataset.iloc[:, -1].values #dependant variable vector
print(X)
print(y)

# ':' means "Range" -iloc

# handle missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')#replace missing values by the mean (Average)
imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3]) # 2nd and 3rd column will be replaced using Transform

print(X)

#encoding categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0] )],remainder='passthrough')
X = np.array(ct.fit_transform(X)) # needs to be a Numpy array for future ML features
print(X)
#encoding for dependant variable (Label Encoder)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y) #set 'y' to be a binary vector 
print(y)


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

print(X_train)
print(X_test)
print(y_train)
print(y_test)

#feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
X_test[:, 3:] = sc.transform(X_test[:, 3:])

print(X_train)
print(X_test)




