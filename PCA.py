import pandas as pd
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load your dataset
data = 'Data/dataset_1.csv'
detectionData = pd.read_csv(data)

# Define columns to drop
columns_to_drop = ["A", "B", "C", "G", "L", "Rf"]

# Drop specified columns
detectionData = detectionData.drop(columns=columns_to_drop)

# Separate features (X) and target variable (y)
X = detectionData.drop("F", axis=1)  # Features
y = detectionData["F"]                # Target variable


# Apply PCA
pca = PCA()
X_pca = pca.fit_transform(X)

# Create a DataFrame with principal components
pc_df = pd.DataFrame(data=X_pca, columns=[f'PC{i}' for i in range(1, X_pca.shape[1] + 1)])
pc_df["F"] = y

# Sort the DataFrame by the 'F' column (0s will be plotted after 1s)
pc_df_sorted = pc_df.sort_values(by='F')

# Create a scatter plot using Plotly Graph Objects
fig = go.Figure()

# Reverse the order of labels
for label in sorted(pc_df_sorted["F"].unique(), reverse=True):
    fig.add_trace(go.Scatter(x=pc_df_sorted[pc_df_sorted["F"] == label]["PC1"],
                             y=pc_df_sorted[pc_df_sorted["F"] == label]["PC2"],
                             mode='markers',
                             name=f'Target (F={label})',
                             marker=dict(size=5)
                             ))

# Update layout for better visualization
fig.update_layout(title='PCA: Scatter Plot of Principal Components (PC1 vs PC2)',
                  xaxis=dict(title='Principal Component 1 (PC1)'),
                  yaxis=dict(title='Principal Component 2 (PC2)'),
                  showlegend=True)

# Show the interactive plot
fig.show()
