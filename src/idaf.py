import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA


class IDAFChallenge:
    def __init__(self, data):
        self.data = data

    def pca_kmeans(self):
        cluster_groups = pd.read_csv(self.data)
        cg = cluster_groups.drop(cluster_groups.columns[[0, 1]], axis=1)
        print(cg.shape)
        #print(cg.describe())
        #print(cg.isnull().sum().sum())
        #print(cg)
        missing_val = [-np.inf, np.inf, "NA", "", None, np.NaN]
        cg.replace(missing_val, np.NaN, inplace=True)
        cg2 = cg.fillna(value=0)
        cg2.isin(missing_val).sum().sum()
        X = cg2
        print(X.shape)
        scaler = StandardScaler()
        scaler.fit(X)
        X_scaled = scaler.transform(X)

        pca_1488 = PCA(n_components=1488, random_state=2022)
        pca_1488.fit(X_scaled)
        x_pca_1488 = pca_1488.transform(X_scaled)
        print(sum(pca_1488.explained_variance_ratio_ * 100))

        plt.plot(np.cumsum(pca_1488.explained_variance_ratio_ * 100))
        plt.xlabel('Number of PCA components')
        plt.ylabel('Explained variance')
        plt.savefig('elbow_plot.png', dpi=100)

        pca_95 = PCA(n_components=0.95, random_state=2022)
        pca_95.fit(X_scaled)
        x_pca_95 = pca_95.transform(X_scaled)
        print("pca",x_pca_95)

        plt.figure(figsize=(10, 7))
        plt.plot(x_pca_95)
        plt.xlabel('Observation')
        plt.ylabel('Transformed data')
        plt.title('Transformed data by the pricipal components(95% variability)', pad=100)
        plt.savefig('pca_95_plot.png')

        # Plot the explained variances
        features = range(pca_95.n_components_)
        plt.bar(features, pca_95.explained_variance_ratio_, color='black')
        plt.xlabel('PCA features')
        plt.ylabel('variance %')
        plt.xticks(features)
        plt.savefig('bar_95_plot.png')

        print("pca", x_pca_95.shape)
        X_new = pd.DataFrame(x_pca_95, columns=['PC' + str(i) for i in range(1, 47)])
        X_train = X_new.copy()
        #print('X_train', X_train)
        #X_new['ref_group'] = X_new['ref_group'].apply(self.ref_group_to_numeric())
        #print(X_new.head())

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

    def ref_group_to_numeric(self, ref):
        if ref == 'group_A':
            return 0
        elif ref == 'group_B':
            return 1
        elif ref == 'group_C':
            return 2
        elif ref == 'group_D':
            return 3
        elif ref == 'group_E':
            return 4
        elif ref == 'group_X':
            return 5
        elif ref == 'unknown':
            return 6

    def main(self):
        self.pca_kmeans()


if __name__ == "__main__":
    data = "../data/raw/sample_data.csv"
    idaf_clustering = IDAFChallenge(data)
    idaf_clustering.main()