import pandas as pd
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load your dataset
data = 'Data/incorrect_predictions.csv'
detectionData = pd.read_csv(data)

# Define columns to drop
columns_to_drop = ["F","L","Rf","FaultyPhase"]

# Drop specified columns
detectionData = detectionData.drop(columns=columns_to_drop)

# Separate features (X) and target variable (y)
X = detectionData.drop(["A","B","C"], axis=1)  # Features
y = detectionData[["A","B","C"]]       # Target variable


# Apply PCA
pca = PCA()
X_pca = pca.fit_transform(X)

# Create a DataFrame with principal components
pc_df = pd.DataFrame(data=X_pca, columns=[f'PC{i}' for i in range(1, X_pca.shape[1] + 1)])

# Add the target variables to the DataFrame
pc_df = pd.concat([pc_df, y.reset_index(drop=True)], axis=1)

# Define the "fig" variable
fig = go.Figure()

# Add traces to the figure for each label
for label in ['A', 'B', 'C']:
    fig.add_trace(go.Scatter(x=pc_df[pc_df[label] == 1]["PC1"],
                             y=pc_df[pc_df[label] == 1]["PC2"],
                             mode='markers',
                             name=f'Target ({label})',
                             marker=dict(size=5)
                             ))

# Update layout for better visualization
fig.update_layout(title='PCA: Scatter Plot of Principal Components (PC1 vs PC2)',
                  xaxis=dict(title='Principal Component 1 (PC1)'),
                  yaxis=dict(title='Principal Component 2 (PC2)'),
                  showlegend=True)

# Show the interactive plot
fig.show()