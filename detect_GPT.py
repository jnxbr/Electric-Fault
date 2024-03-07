import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from keras.layers import Dense
from keras.models import Sequential

# Lee el archivo CSV
detectionData_pre = pd.read_csv('Data/dataset_1.csv')

# Filtra los datos
detectionData = detectionData_pre[detectionData_pre["Rf"] != 25]

# Define las columnas de entrada (X_detection) y salida (y_detection)
input_columns = ["Ia", "Ib", "Ic", "Va", "Vb", "Vc"]
output_column = "F"

X_detection = detectionData[input_columns].values
y_detection = detectionData[output_column].values

# Guarda los datos originales para análisis posterior
additional_columns = ["A", "B", "C", "G", "L", "Rf"]
additional_data = detectionData[additional_columns]

# Divide los datos en conjuntos de entrenamiento y prueba
Xd_train, Xd_test, yd_train, yd_test, additional_data_train, additional_data_test = train_test_split(X_detection, y_detection, additional_data, test_size=0.2, random_state=0)

# Crear y compilar el modelo de la red neuronal
detectionANN = Sequential()
detectionANN.add(Dense(units=7, activation='relu'))
detectionANN.add(Dense(units=5, activation='relu'))
detectionANN.add(Dense(units=3, activation='relu'))
detectionANN.add(Dense(units=1, activation='sigmoid'))
detectionANN.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
detectionANN.fit(Xd_train, yd_train, batch_size=32, epochs=100)

# Realizar predicciones en el conjunto de prueba
yd_pred = detectionANN.predict(Xd_test)
yd_pred = (yd_pred > 0.5)

# Comparar predicciones con datos reales
predictions_vs_actual = pd.DataFrame({'Predicted': yd_pred.flatten(), 'Actual': yd_test.flatten()})


# Mostrar información sobre predicciones incorrectas
print("Información sobre predicciones incorrectas:")
print(incorrect_predictions)

# Matriz de confusión y precisión
cm = confusion_matrix(yd_test, yd_pred)
acc = accuracy_score(yd_test, yd_pred)
tn, fp, fn, tp = cm.ravel()

print("\nFP")
print(fp)
print("\nFN")
print(fn)

print("\nMatriz de confusión:")
print(cm)

print(f"\nCorrect Predictions : {tn + tp}")
print(f"Wrong Predictions : {fn + fp}")
print(f"Accuracy : {acc * 100} %")

# Guardar el modelo entrenado
detectionANN.save('Model/detection_model.h5')

incorrect_predictions.to_csv('Data/incorrect_predictions.csv', index=False)