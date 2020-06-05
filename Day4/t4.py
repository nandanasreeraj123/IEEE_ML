import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
data = pd.read_csv("http://iali.in/datasets/IEEEAPSIT/unsupervised-ml/Wholesale%20customers%20data.csv")

#to see if we are correct
indices = [14,74,228,300,310]

ind = pd.DataFrame(data.loc[indices], columns=data.keys()).reset_index(drop=True)
cov_to_log = np.log(data.copy())
converted_samples = np.log(ind)

for every in cov_to_log.keys():
    Q1 = np.percentile(cov_to_log, 25)
    Q3 = np.percentile(cov_to_log, 75)
    step = (Q3 - Q1) * 1.5
    print("outliners '{}':".format(every))
    print(cov_to_log[~((cov_to_log[every] >= Q1 - step) & (cov_to_log[every] <= Q3 + step))])

pca = PCA(n_components=6)
pca.fit(cov_to_log)
pca_samples = pca.transform(converted_samples)
print(pca.components_)
print(pca.explained_variance_)
print(pca_samples)
