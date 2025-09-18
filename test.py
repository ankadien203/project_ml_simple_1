import pickle
import pandas as pd
from sklearn.model_selection import  train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from diabetes import model

data_2 = pd.read_csv('diabetes.csv')
name_column = 'Outcome'
x = data_2.drop(name_column, axis = 1)
y = data_2[name_column]

x_train_2, x_test_2, y_train_2, y_test_2 = train_test_split(x,y, test_size= 0.2, random_state= 66)
scaler = StandardScaler()
x_test_2_chuan_hoa = scaler.fit(x_test_2)

predict_2 = model.predict(x_test_2_chuan_hoa)
print(accuracy_score(y_test_2, predict_2))

