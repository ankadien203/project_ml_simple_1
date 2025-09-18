#step by step to code a simple machine learning project
#step 1: split input and output in data
#step 2: split train and test in data
#step 3: preprocessing in train data, use standard scaler to find mean and std
#step 4: choose model to train and save this model to a .sav file
#step 5: prediction

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, scale
from sklearn.svm import  SVC
import pickle
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

data = pd.read_csv('diabetes.csv')
print(data)
output_name = 'Outcome'
x = data.drop(output_name, axis = 1)
y = data[output_name]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, random_state= 42)

scaler = StandardScaler()
x_train_chuan_hoa = scaler.fit_transform(x_train)
x_test_chuan_hoa = scaler.transform(x_test)

model = SVC()
model.fit(x_train_chuan_hoa, y_train)
with open('model.sav','wb') as f:
    pickle.dump(model, f)

predict = model.predict(x_test_chuan_hoa)
print(f1_score(y_test, predict))




