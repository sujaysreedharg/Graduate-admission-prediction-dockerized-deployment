
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
Reading = pd.read_csv('Admission_Predict.csv')


Reading = Reading.rename(columns={'GRE Score': 'GRE Score', 'TOEFL Score': 'TOEFL Score', 'LOR ': 'LOR', 'Chance of Admit ': 'Admit Possibilty'})


Reading.drop('Serial No.', axis='columns', inplace=True)




x=Reading.drop('Admit Possibilty',axis='columns')
y=Reading['Admit Possibilty']
x_train,x_test,y_train,y_test=train_test_split(x, y)


linear_regression = LinearRegression()
linear_regression = linear_regression.fit(x_train,y_train)
model = LinearRegression(normalize=True)
model.fit(x_test, y_test)


filename = 'linearregressionmodel.pkl'
pickle.dump(model, open(filename, 'wb'))