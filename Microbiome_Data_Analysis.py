import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Function for ComBat algorithm
def combat(data, batch):
    # Compute overall mean and variance
    overall_mean = data.mean(axis=0)
    overall_var = data.var(axis=0)
    
    # Compute batch means and variances
    batch_means = data.groupby(batch).mean()
    batch_vars = data.groupby(batch).var()
    
    # Compute weights
    weights = data.groupby(batch).count()
    
    # Compute gamma star and delta star squared
    gamma_star = np.sqrt(batch_vars.mean(axis=0))
    delta_star2 = (weights * (batch_means - overall_mean)**2).sum(axis=0) / (weights.sum(axis=0) - len(weights))
    
    # Compute lambda star squared
    m_value = (1 + delta_star2) * ((weights * (batch_means - overall_mean)**2).sum(axis=0) / (gamma_star**2))
    lambda_star2 = m_value / (m_value + data.shape[0] - 1) * (1 - delta_star2 + delta_star2 / gamma_star)
    
    # Compute beta star and alpha star
    beta_star = lambda_star2 / gamma_star * overall_var
    alpha_star = overall_mean - beta_star * overall_mean
    
    # Correct data
    corrected_data = data.copy()
    for b in batch.unique():
        corrected_data[batch == b] = (data[batch == b] - alpha_star) / np.sqrt(beta_star)
        corrected_data[batch == b] = corrected_data[batch == b] * np.sqrt(gamma_star) + overall_mean
    
    return corrected_data

# Load the data
data = pd.read_csv('clr_transformed_data.csv')

# Perform PCA
pca_model = PCA(n_components=2)
principal_components = pca_model.fit_transform(data.drop(['sampleID', 'Date'], axis=1))

# Visualize PCA before correction
pca_df = pd.DataFrame(data=principal_components, columns=['Principal Component 1', 'Principal Component 2'])
pca_df['Batch'] = data['Date']
sns.scatterplot(x='Principal Component 1', y='Principal Component 2', hue='Batch', data=pca_df)
plt.title('PCA Before Correction')
plt.show()

# Perform ComBat algorithm
corrected_data = combat(data.drop(['sampleID', 'Date'], axis=1), data['Date'])

# Perform PCA on corrected data
principal_components_corrected = pca_model.fit_transform(corrected_data)

# Visualize PCA after correction
pca_corrected_df = pd.DataFrame(data=principal_components_corrected, columns=['Principal Component 1', 'Principal Component 2'])
pca_corrected_df['Batch'] = data['Date']
sns.scatterplot(x='Principal Component 1', y='Principal Component 2', hue='Batch', data=pca_corrected_df)
plt.title('PCA After Correction')
plt.show()
