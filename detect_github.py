import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from keras import utils
from keras.layers import Dense
from keras.models import Sequential

detectionData = pd.read_csv('Data/detect_dataset.csv')
detectionData.drop(['Unnamed: 7', 'Unnamed: 8'], axis='columns' ,inplace=True)

Xdetection = detectionData.iloc[:, 1:].values
ydetection = detectionData.iloc[:, 0].values

Xd_train, Xd_test, yd_train, yd_test = train_test_split(Xdetection, ydetection, test_size = 0.2, random_state = 0)

sc = StandardScaler()
Xd_train = sc.fit_transform(Xd_train)
Xd_test = sc.transform(Xd_test)

detectionANN = Sequential()
detectionANN.add(Dense(units=6, activation='relu'))
detectionANN.add(Dense(units=6, activation='relu'))
detectionANN.add(Dense(units=1, activation='sigmoid'))
detectionANN.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

detectionANN.fit(Xd_train, yd_train, batch_size=20, epochs=30)

yd_pred = detectionANN.predict(Xd_test)
yd_pred = (yd_pred > 0.5)

cm = confusion_matrix(yd_test, yd_pred)
acc = accuracy_score(yd_test, yd_pred)
tn, fp, fn, tp = cm.ravel()
print(f"Correct Predictions : {tn + tp}")
print(f"Wrong Predictions : {fn + fp}")
print(f"Accuracy : {acc * 100} %")

detectionANN.save("Model/detectionANN.h5")
