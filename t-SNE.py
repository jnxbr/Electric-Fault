import pandas as pd
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

# Load your dataset
data = 'Data/dataset_4.csv'
detectionData = pd.read_csv(data)

# Define columns to drop
columns_to_drop = ["A", "B", "C", "G", "L", "Rf"]

# Drop specified columns
detectionData = detectionData.drop(columns=columns_to_drop)

# Separate features (X) and target variable (y)
X = detectionData.drop("F", axis=1)  # Features
y = detectionData["F"]                # Target variable

# Standardize the features
scaler = StandardScaler()
X_standardized = scaler.fit_transform(X)

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_standardized)

# Create a DataFrame with t-SNE components
tsne_df = pd.DataFrame(data=X_tsne, columns=['TSNE1', 'TSNE2'])
tsne_df["F"] = y

# Create a scatter plot using Plotly Graph Objects
fig = go.Figure()

for label in tsne_df["F"].unique():
    fig.add_trace(go.Scatter(x=tsne_df[tsne_df["F"] == label]["TSNE1"],
                             y=tsne_df[tsne_df["F"] == label]["TSNE2"],
                             mode='markers',
                             name=f'Target (F={label})',
                             marker=dict(size=5)
                             ))

# Update layout for better visualization
fig.update_layout(title='t-SNE: Scatter Plot of t-SNE Components',
                  xaxis=dict(title='t-SNE Component 1'),
                  yaxis=dict(title='t-SNE Component 2'),
                  showlegend=True)

# Show the interactive plot
fig.show()
