import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
data = pd.read_csv("http://iali.in/datasets/IEEEAPSIT/unsupervised-ml/Wholesale%20customers%20data.csv")
#to see if we are correct
indices = [14,74,228,300,310]

samples = pd.DataFrame(data.loc[indices], columns=data.keys()).reset_index(drop=True)
log_data = np.log(data.copy())
log_samples = np.log(samples)
print("Sample:")
print(samples)
print("Log-transformed samples:")
print(log_samples)
for feature in log_data.keys():
    Q1 = np.percentile(log_data, 25)
    Q3 = np.percentile(log_data, 75)
    step = (Q3 - Q1) * 1.5
    print("Data points considered outliers for the feature '{}':".format(feature))
    print(log_data[~((log_data[feature] >= Q1 - step) & (log_data[feature] <= Q3 + step))])

pca = PCA(n_components=6)
pca.fit(log_data)
pca_samples = pca.transform(log_samples)
print(pca.components_)
print(pca.explained_variance_)
print(pca_samples)
