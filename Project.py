import pandas
import numpy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt  
#read file
csv = pandas.read_csv("Data.csv")

# Dropping any rows with Nan values 
csv.dropna(inplace = True) 

#values that prediction is based on 
x = csv[["AirTemp", "Press", "UMR"]]

#values to be predicted
y = csv[["NO", "NO2", "O3", "PM10"]]

#return four marix: two for learn and two for test
x_learn, x_test, y_learn, y_test = train_test_split(
                                x, y, test_size=0.3, random_state=0)
 
#transformer for transforming the values
transformer = RobustScaler().fit(x_learn)
#scalar type of the x_learn matrix
x_learn_scalar = transformer.transform(x_learn)
#scalar type of the x_test matrix
x_test_scalar = transformer.transform(x_test)

model = LinearRegression(fit_intercept=True, normalize=True).fit(x_learn_scalar, y_learn)
#returns coefficient of determination
determ_coef = model.score(x_test_scalar, y_test)
#returns the intercept for each value
intercept =  model.intercept_
#returns the slope for each value 
slope =  model.coef_

print("Coefficient of determination: ", determ_coef)
print("Intercept: ", intercept)
print("Slope: ", slope)

a = numpy.array([[-1.4, 941, 73.7]])
a_transformed = transformer.transform(a)
prediction= model.predict(a_transformed)
print("Values for AirTemp, Press, и UMR\n", a)
print("Prediction:\n", prediction)



