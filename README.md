# Chainlearn

Mini module with some syntax sugar utilities for pandas and sklearn. It basically allows you turn this:

```python
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE 
from sklearn.cluster import KMeans
 
iris = sns.load_dataset('iris').drop('species', axis=1)
 
pca = PCA(n_components=3)
tsne = TSNE(n_components=2)

kmeans = KMeans(n_clusters=2)

transformed = tsne.fit_transform(pca.fit_transform(iris))

cluster_labels = kmeans.fit_predict(transformed)

plt.scatter(transformed[:, 0], transformed[:, 1], c=cluster_labels)
```

Into a chainlearn pipeline that tries to look like a "tidyverse" version:

```python
import seaborn as sns
import chainlearn
from matplotlib import pyplot as plt

iris = sns.load_dataset('iris')

(iris
 .drop('species', axis=1)
 .learn.PCA(n_components=3)
 .learn.TSNE(n_components=2)
 .assign(
     cluster=lambda df: df.learn.KMeans(n_clusters=2)
 )
 .plot
 .scatter(
     x=0,
     y=1,
     c='cluster',
     cmap=plt.get_cmap('viridis')
 )
);
```

This is achieved by attaching some sklearn model and preprocessing classes to the pandas `DataFrame` and `Series` classes, and trying to guess what methods should be called. 

You can also do supervised/regressions/etc:

```python
(iris
 .assign(
     species=lambda df: df['species'].learn.LabelEncoder()
 )
 .learn.RandomForestClassifier(
     n_estimators=100,
     target='species'
 )
 .rename(columns={0: 'label'})
 .plot
 .hist()
)
```

Check out the [examples notebook]('./examples/showcase.ipynb')...

## Other stuff you can do 

Additionally, there are a couple of methods you can call to shorten some tasks.

### Explain

Calling `explain` at the end of your chainlearn pipeline will get you whatever the model has to try to explain itself. In linear models this will be the coefficients, while ensemble models will have feature importances (in sklearn computed as mean decrease impurity for most models).

```python
(iris
 .assign(
     species=lambda df: df['species'].learn.LabelEncoder()
 )
 .learn.Lasso(alpha=0.01, target='species')
 .learn.explain()
 .plot
 .bar()
);
```

I may add some SHAP value calculations in the near future.

### Cross-validate

There is also a `cross_validate` function that will perform cross validation and get you the scores.

```python
(iris
 .assign(
     species=lambda df: df['species'].learn.LabelEncoder()
 )
 .learn.RandomForestClassifier(
     n_estimators=100,
     target='species'
 )
 .learn.cross_validate(folds=5, scoring='f1_macro')
 .plot
 .hist()
);
```

## Attaching your own models

If you have your own module with models that follow the sklearn api (i.e. have `fit` and/or `fit_predict`, `fit_transform`, `transform`, `predict` methods) you can attach them to `DataFrames` and `Series`:

```python
import mymodels # Contains a MyModel class with a fit_transform method
from chainlearn import attach
attach(mymodels)

(iris
 .learn.MyModel(params=params)
 .plot
 .scatter(x=0, y=1)
);
```


## Install

`pip install chainlearn` or install locally by cloning, changing to the repo dir and `pip install -e .`
