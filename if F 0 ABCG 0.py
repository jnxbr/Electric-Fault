import pandas as pd

# Lee el CSV original
data = 'Data/dataset_3.csv'
detectionData = pd.read_csv(data)

# Aplicar la condición y actualizar los valores
detectionData.loc[detectionData['F'] == 0, ['A', 'B', 'C', 'G']] = 0

# Sobrescribe el archivo CSV original con el DataFrame modificado
detectionData.to_csv(data, index=False)
