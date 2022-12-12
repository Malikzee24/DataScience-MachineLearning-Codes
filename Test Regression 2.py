import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error
 # Only checking Mean Squared Error, Coeffiecent and Intercept values to determine accuracy of model:

diabetes = datasets.load_diabetes()

# Printing Data from key in coloumn of indexing number 2:
diabetes_X = diabetes.data
#print(diabetes_X)

# Training and Testing of data by slicing of X-axis:
diabetes_X_train = diabetes_X[:-30]
diabetes_X_test = diabetes_X[-20:]

# Training and Testing of data by slicing of Y-axis:
diabetes_Y_train = diabetes.target[:-30]
diabetes_Y_test = diabetes.target[-20:]

# Now we use linear model regression:
model = linear_model.LinearRegression()

# Fitting The Data : it means with help of data we create a line and that line will save in linear model known as "Fitting The Data"
# 1st: Fitting Data Line in Training of X & Y data:
model.fit(diabetes_X_train, diabetes_Y_train)

#2nd: Predicting the Data by Testing of X & Y data: (Note: Predicting the data gives right, wrong, less acurate or almost close value)
# (Note: Predicting the data gives right, wrong, less acurate or almost close value. It depends given features of data)
diabetes_Y_predicted = model.predict(diabetes_X_test)

# Print mean squared error by help of mean_squared_error library via sklearn.
print("Mean Squared Error is:", mean_squared_error(diabetes_Y_test, diabetes_Y_predicted))

# Printing model coefficient and model intercept values:
print("Weights", model.coef_)
print("Intercept", model.intercept_)


# Result is much accurate then previous Test Regression.py file. So Test Regression2.py is more accurate due to adding all features of data.

#Output is :
#Mean Squared Error is: 2004.2629212944946
#Weights [  -1.16678648 -237.18123633  518.31283524  309.04204042 -763.10835067  458.88378916   80.61107395  174.31796962  721.48087773   79.1952801 ]
#Intercept 153.05824267739402
