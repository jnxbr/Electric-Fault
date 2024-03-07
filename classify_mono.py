import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from keras import utils
from keras.layers import Dense
from keras.models import Sequential
import plotly.express as px
import seaborn as sns

data = 'Data/dataset_4.csv'
classificationData = pd.read_csv(data)
classificationData = classificationData[classificationData['F'] == 1]
# classificationData['FaultyPhase'] = np.where((classificationData['A'] == 1) & (classificationData['B'] == 0) & (classificationData['C'] == 0), 'A',
#                                 np.where((classificationData['A'] == 0) & (classificationData['B'] == 1) & (classificationData['C'] == 0), 'B',
#                                 np.where((classificationData['A'] == 0) & (classificationData['B'] == 0) & (classificationData['C'] == 1), 'C', 'None')))
classificationData['FaultyPhase'] = np.where((classificationData['A'] == 1) & (classificationData['B'] == 0) & (classificationData['C'] == 0), 1, 0)

multi_phase_indices = classificationData[classificationData['FaultyPhase'] == 'None'].index
classificationData.loc[multi_phase_indices, ['A', 'B', 'C']] = 0

X_columns = ["Ia","Ib", "Ic" ,"Va", "Vb" ,"Vc"]
X_classification = classificationData[X_columns]
y_columns = ["FaultyPhase"]
y_classification = classificationData[y_columns]

X_train, X_test, y_train, y_test = train_test_split(X_classification, y_classification, test_size=0.2, random_state=0)

classificationANN = Sequential()
classificationANN.add(Dense(units=10, activation='relu'))
classificationANN.add(Dense(units=1, activation='sigmoid'))
classificationANN.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

classificationANN.fit(X_train, y_train, batch_size=800, epochs=30)

y_pred = classificationANN.predict(X_test)
y_pred = (y_pred > 0.5)


loss, acc = classificationANN.evaluate(X_test, y_test)
print(f'Accuracy : {acc*100} %')


classificationANN.save("Model/classification_model_mono.h5")

# Test en todos los datos ordenados
y_pred = classificationANN.predict(X_classification)
y_pred = (y_pred > 0.5)

incorrect_positions = np.where(y_pred != y_classification)[0]

incorrect_predictions = classificationData.iloc[incorrect_positions,:]

incorrect_predictions.to_csv('Data/incorrect_predictions.csv', index=False)
