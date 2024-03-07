# Import necessary libraries
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
import plotly.express as px

# Define the path to the dataset
data = 'Data/dataset_4.csv'

# Load the dataset into a pandas DataFrame
detectionData = pd.read_csv(data)

# Define the columns to be used as features (X)
X_columns = ["Ia","Ib", "Ic" ,"Va", "Vb" ,"Vc"]
X_detection = detectionData[X_columns]

# Define the column to be used as the target (y)
y_columns = ["F"]
y_detection = detectionData[y_columns]

# Define additional columns to be used for further analysis
additional_columns = ["A", "B", "C", "G", "L", "Rf"]
additional_data = detectionData[additional_columns]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test, additional_data_train, additional_data_test = train_test_split(X_detection, y_detection, additional_data, test_size=0.2, random_state=0)

# Uncomment to Standardize the features
# sc = StandardScaler()
# Xd_train = sc.fit_transform(Xd_train)
# Xd_test = sc.transform(Xd_test)

# Initialize a Sequential model
detectionANN = Sequential()

detectionANN.add(Dense(units=6, activation='relu'))
detectionANN.add(Dense(units=1, activation='sigmoid'))
detectionANN.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Train the ANN using the training data
detectionANN.fit(X_train, y_train, batch_size=400, epochs=50)

# Print a summary of the model structure
detectionANN.summary()

# Use the trained ANN to predict the test data
y_pred = detectionANN.predict(X_test)

# Convert the predicted probabilities to binary outputs
y_pred = (y_pred > 0.5)

# Compute the confusion matrix and accuracy score of the predictions
cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)

# Extract the individual components of the confusion matrix
tn, fp, fn, tp = cm.ravel()

# Print the number of correct and wrong predictions, and the accuracy
print(f"Correct Predictions : {tn + tp}")
print(f"Wrong Predictions : {fn + fp}")
print(f"Accuracy : {acc * 100} %")

# Use the trained ANN to predict the test data
y_pred = detectionANN.predict(X_detection)

# Convert the predicted probabilities to binary outputs
y_pred = (y_pred > 0.5)

# Find the positions where the predictions were incorrect
incorrect_positions = np.where(y_pred != y_detection)[0]

# Get the rows of the data where the predictions were incorrect
incorrect_predictions = detectionData.iloc[incorrect_positions,:]