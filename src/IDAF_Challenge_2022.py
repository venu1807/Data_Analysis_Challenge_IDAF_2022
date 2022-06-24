#!/usr/bin/env python
# coding: utf-8

# In[127]:


import os
cwd = os.getcwd()
print(cwd)


# In[128]:


import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np

#pd.set_option('display.max_columns', None)
#pd.set_option('display.max_rows', None)

get_ipython().run_line_magic('matplotlib', 'inline')


# In[129]:


cluster_groups = pd.read_csv('/home/venugopalachhe/Desktop/Data_Analysis_Challenge_IDAF_2022/data/raw/sample_data.csv')


# In[130]:


cluster_groups.head()


# In[131]:



cluster_groups['ref_group'].unique()


# In[132]:


cluster_groups['feature_1'].unique()


# In[133]:


cg = cluster_groups.drop(cluster_groups.columns[[0,1]], axis=1)


# In[134]:


cg.head()


# # (There are two common ways to deal with high dimensional data:)
# ## A.Choose to include fewer features. Ways to decide which features to drop from the sample dataset
# ### 1. Drop features with many missing values
# ### 2. Drop features with low variance
# ### 3. Drop features with low correlation with the response variable
# 
# ## B.Use a regularization method.
# ### 1. Principal Components Anlysis
# ### 2. Pricipal Components Regression
# ### 3. Lasso Regression
# ### 4. Ridge Regression

# In[135]:


cg.describe()


# # Handling Missing Values in the Dataset
# 
# ## 1. Either fill the null values with some values
# ## 2. Drop such rows
# ## 3. Replace them

# In[136]:


#pd.set_option('display.max_rows', None)
cg.isnull().sum().sum()


# In[137]:


missing_val = [-np.inf, np.inf, "NA", "", None, np.NaN]

cg.isin(missing_val).sum().sum()


# In[138]:


cg.replace(missing_val, np.NaN, inplace = True)


# In[139]:


#pd.set_option('display.max_rows', None)
cg.isnull().sum().sum()


# In[140]:


cg2 = cg.fillna(value = 0)


# In[141]:


cg2.isnull().sum().sum()


# In[142]:


cg2.isin(missing_val).sum().sum()


# In[143]:


cg2.shape


# In[144]:


cg2.head()


# In[145]:


X = cg2.values
X.shape


# In[146]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X)


# In[147]:


X_scaled = scaler.transform(X)


# In[148]:


from sklearn.decomposition import PCA


# In[149]:


pca_1488 = PCA(n_components = 1488, random_state = 2022)
pca_1488.fit(X_scaled)
x_pca_1488= pca_1488.transform(X_scaled)


# In[150]:


print(sum(pca_1488.explained_variance_ratio_ * 100))


# In[151]:


pca_1488.explained_variance_ratio_ * 100


# In[152]:


variance_features = np.cumsum(pca_1488.explained_variance_ratio_ * 100)

for index, var in enumerate(variance_features):
    if var > 99:
        print(f"sufficient no of pca components: {index}, covered variance: {var}")
        break
    else:
        print(index, var)


# In[153]:


plt.plot(np.cumsum(pca_1488.explained_variance_ratio_*100))
plt.xlabel('Number of components')
plt.ylabel('Explained variance')
plt.savefig('elbow_plot.png', dpi=100)


# In[154]:


pca_120 = PCA(n_components = 120, random_state = 2022)
pca_120.fit(X_scaled)
x_pca_120= pca_120.transform(X_scaled)


# # Apply PCA by setting n_components = 0.95

# In[155]:


pca_95 = PCA(n_components = 0.95, random_state = 2022)
pca_95.fit(X_scaled)
x_pca_95 = pca_95.transform(X_scaled)


# In[156]:


x_pca_95.shape


# In[157]:


pca_99 = PCA(n_components = 0.99, random_state = 2022)
pca_99.fit(X_scaled)
x_pca_99 = pca_99.transform(X_scaled)


# In[158]:


x_pca_99.shape


# In[159]:


plt.figure(figsize=(10,7))
plt.plot(x_pca_95)
plt.xlabel('Observation')
plt.ylabel('Transformed data')
plt.title('Transformed data by the pricipal components(95% variability)',pad=100)
plt.savefig('pca_95_plot.png')


# In[172]:


# Plot the explained variances
features = range(pca_95.n_components_)
plt.bar(features, pca_95.explained_variance_ratio_, color='black')
plt.xlabel('PCA features')
plt.ylabel('variance %')
plt.xticks(features)


# In[173]:


plt.scatter(x_pca_95[0], x_pca_95[1], alpha=.1, color='black')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')


# In[160]:


X_new = pd.DataFrame(x_pca_95, columns = ['PC'+ str(i) for i in range(1, 47)])


# In[161]:


X_train = X_new.copy()


# In[162]:


X_new['ref_group'] = cluster_groups['ref_group']


# In[163]:


X_new.head()


# In[164]:


def ref_group_to_numeric(ref):
    if ref == 'group_A': return 0
    elif ref == 'group_B':   return 1
    elif ref == 'group_C':   return 2
    elif ref == 'group_D':   return 3
    elif ref == 'group_E':   return 4
    elif ref == 'group_X':   return 5
    elif ref == 'unknown':   return 6


# In[165]:


X_new['ref_group'] = X_new['ref_group'].apply(ref_group_to_numeric)


# In[169]:


X_new.head()


# In[167]:


#X_new.to_csv('/home/venugopalachhe/Desktop/Data_Analysis_Challenge_IDAF_2022/data/processed/'+ 'idaf_processed_data.csv', index = False)


# # Now, the data is being reduced into principal components for clustering into groups (PCA + Clustering Method)

# # 1. K-Means Clustering

# In[ ]:


from sklearn.cluster import KMeans


# In[179]:


ks = range(1, 14)
inertias = []
for k in ks:
    # Create a KMeans instance with k clusters: model
    model = KMeans(n_clusters=k)
    
    # Fit model to samples
    model.fit(X_train)
    
    # Append the inertia to the list of inertias
    inertias.append(model.inertia_)
    
plt.plot(ks, inertias, '-o', color='black')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(ks)
plt.title('plot showing a slow decrease of inertia after clusters(k) = 7')
plt.savefig('kmeans_elbow_plot.png')


# In[ ]:




