import numpy as np
import seaborn as sns
import chainlearn
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import unittest

iris = sns.load_dataset('iris')
iris_no_species = iris.drop('species', axis=1)


class TestCompareToVanillaSklearn(unittest.TestCase):

    def test_pca_kmeans_pipeline(self):

        rs = np.random.RandomState(1234)

        pca = PCA(n_components=3, random_state=rs)
        kmeans = KMeans(n_clusters=2, random_state=rs)

        transformed = pca.fit_transform(iris_no_species)

        cluster_labels = kmeans.fit_predict(transformed)

        rs = np.random.RandomState(1234)
        chainlearn_cluster_labels = (iris
                 .drop('species', axis=1)
                 .learn.PCA(n_components=3, random_state=rs)
                 .assign(
                     cluster=lambda df: df.learn.KMeans(n_clusters=2, random_state=rs)
                 )
         )['cluster'].values

        self.assertEqual(chainlearn_cluster_labels.tolist(), cluster_labels.tolist())

    def test_classification(self):
        rs = np.random.RandomState(1234)
        rf = RandomForestClassifier(random_state=rs)
        rf.fit(iris_no_species, LabelEncoder().fit_transform(iris['species']))
        res = rf.predict(iris_no_species).ravel()

        rs = np.random.RandomState(1234)
        chainlearn_res = (iris
                 .assign(
                     species=lambda df: df['species'].learn.LabelEncoder()
                 )
                 .learn.RandomForestClassifier(
                     random_state=rs,
                     target='species'
                 )
             ).values.ravel()

        self.assertEqual(chainlearn_res.tolist(), res.tolist())


if __name__ == '__main__':
    unittest.main()
