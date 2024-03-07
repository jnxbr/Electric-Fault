# Import necessary libraries
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

# Define the path to the dataset
data = 'Data/dataset_5.csv'

# Load the dataset into a pandas DataFrame
classificationData = pd.read_csv(data)

# Filter the data to include only the rows where 'F' is 1
classificationData = classificationData[classificationData['F'] == 1]

# Create a new column 'FaultyPhase' based on the values of columns 'A', 'B', and 'C'
classificationData['FaultyPhase'] = np.where((classificationData['A'] == 1) & (classificationData['B'] == 0) & (classificationData['C'] == 0), 'A',
                                np.where((classificationData['A'] == 0) & (classificationData['B'] == 1) & (classificationData['C'] == 0), 'B',
                                np.where((classificationData['A'] == 0) & (classificationData['B'] == 0) & (classificationData['C'] == 1), 'C', 'None')))

# Define the columns to be used as features (X)
X_columns = ["Ia","Ib", "Ic" ,"Va", "Vb" ,"Vc"]
X_classification = classificationData[X_columns]

# Define the columns to be used as the target (y)
y_columns = ["A", "B", "C","G"]
y_classification = classificationData[y_columns]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_classification, y_classification, test_size=0.2, random_state=0)

# Initialize a Sequential model
classificationANN = Sequential()
classificationANN.add(Dense(units=10, activation='relu'))
classificationANN.add(Dense(units=4, activation='sigmoid'))
classificationANN.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Train the ANN using the training data
classificationANN.fit(X_train, y_train, batch_size=100, epochs=50)

# Use the trained ANN to predict the test data
y_pred = classificationANN.predict(X_test)

# Convert the predicted probabilities to binary outputs
y_pred = (y_pred > 0.5)

# Compute the multilabel confusion matrix and accuracy score of the predictions
cm = multilabel_confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)

print(y_test)
print(y_pred)

# Create a figure and a grid of subplots
fig, axes = plt.subplots(nrows=1, ncols=len(cm))

# For each class
for i, matrix in enumerate(cm):
    sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', ax=axes[i])
    axes[i].set_title(f'Confusion Matrix for Class {i}')
    axes[i].set_xlabel('Predicted Label')
    axes[i].set_ylabel('True Label')

# Display the figure with all the subplots
plt.tight_layout()
plt.show(block=False)

# Print the confusion matrix and accuracy
print(cm)
print(f"Accuracy : {acc * 100} %")

# Save the trained ANN model
classificationANN.save("Model/classificationANN.h5")

# Test the trained ANN on all the data
y_pred = classificationANN.predict(X_classification)

# Convert the predicted probabilities to binary outputs
y_pred = (y_pred > 0.5)

# Compute the multilabel confusion matrix and accuracy score of the predictions
cm = multilabel_confusion_matrix(y_classification, y_pred)
acc = accuracy_score(y_classification, y_pred)

# Create a figure and a grid of subplots
fig, axes = plt.subplots(nrows=1, ncols=len(cm))

# For each class
for i, matrix in enumerate(cm):
    sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', ax=axes[i])
    axes[i].set_title(f'Confusion Matrix for Class {i}')
    axes[i].set_xlabel('Predicted Label')
    axes[i].set_ylabel('True Label')

# Display the figure with all the subplots
plt.tight_layout()
plt.show()

# Print the confusion matrix and accuracy
print(cm)
print(f"Accuracy : {acc * 100} %")

# Find the positions where the predictions were incorrect
incorrect_positions = np.where(y_pred != y_classification)[0]

# Get the rows of the data where the predictions were incorrect
incorrect_predictions = classificationData.iloc[incorrect_positions,:]
incorrect_predictions.to_csv('Data/incorrect_predictions.csv', index=False)
