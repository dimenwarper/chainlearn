from abc import ABCMeta

import numpy as  np

import warnings
import pandas as pd

from sklearn.model_selection import cross_val_score
import sklearn.linear_model
import sklearn.cluster
import sklearn.decomposition
import sklearn.ensemble
import sklearn.preprocessing
import sklearn.manifold

from typing import Union, Optional


__METHODS: str =  ['predict', 'transform', 'fit_predict', 'fit_transform']

@pd.api.extensions.register_dataframe_accessor('learn')
@pd.api.extensions.register_series_accessor('learn')
class LearnAccessor(object):
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

def __register_chain_vars(
        df: pd.DataFrame,
        **kwargs
        ):
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=UserWarning)
        if hasattr(df, '_chain_vars'):
            setattr(
                    df,
                    '_chain_vars',
                    getattr(
                        df,
                        '_chain_vars'
                        ).update(**kwargs)
                    )
        else:
            setattr(
                    df,
                    '_chain_vars',
                    kwargs
                    )

def __get_chain_vars(df: pd.DataFrame):
    return getattr(df, '_chain_vars')

def __check_if_class_can_be_attached(cls):
    valid_class: bool = type(cls) in [ABCMeta, type]

    has_valid_method: bool = False
    for method in __METHODS:
        if hasattr(cls, method):
            has_valid_method = True
            break

    return valid_class & has_valid_method

def get_models_from_module(module):
    models = []
    for name in module.__all__:
        cls = module.__dict__.get(name)
        if __check_if_class_can_be_attached(cls):
            models.append(cls)
    return models

def generate_model_function(model_class: Union[ABCMeta, type]):
    method: str = ''
    for method in __METHODS:
        if hasattr(model_class, method):
            break

    def model_fun(
            self: LearnAccessor,
            target: Optional[Union[str, pd.DataFrame, pd.Series, np.array]] = None,
            *args,
            **kwargs
            ):
        model = model_class(*args, **kwargs)

        if type(self._obj) == pd.DataFrame and target in self._obj.columns:
            Y = self._obj[target]
            X = self._obj.drop(target, axis=1)
        else:
            Y = target
            X = self._obj
        if type(X) == pd.DataFrame and len(X.shape) == 1:
            X = X.reshape(len(X), 1)

        if method in ['predict', 'transform']:
            if Y is None:
                model.fit(X)
            else:
                model.fit(X, Y)

        res = getattr(model, method)(X)

        if type(res) == pd.DataFrame:
            return res
        else:
            df = pd.DataFrame(
                    res,
                    index=self._obj.index
                    )
            __register_chain_vars(
                    df,
                    model=model,
                    prev=X,
                    target=Y
                    )
            return df
    return model_fun

def generate_explain_function():
    def explain_fun(self: LearnAccessor):
        __chain_vars = __get_chain_vars(self._obj)
        if 'sklearn.ensemble' in __chain_vars['model'].__module__:
            res = __chain_vars['model'].feature_importances_
            col_prefix = 'feature_importance'
        else:
            if hasattr(__chain_vars['model'], 'coef_'):
                res = __chain_vars['model'].coef_
                col_prefix = 'coefficient'
            else:
                raise ValueError(f'Models of module {self.__module__} not fully supported yet')

        if len(res.shape) == 1:
            res = res.reshape(len(res), 1)

        return pd.DataFrame(
                res,
                columns=[f'{col_prefix}_{i}' for i in range(res.shape[1])],
                index=__chain_vars['prev'].columns
                )
    return explain_fun

def generate_cross_validate_function():
    def cross_validate_fun(
            self: LearnAccessor,
            folds: Union[int, float],
            **cvargs
            ):
        __chain_vars = __get_chain_vars(self._obj)
        res = cross_val_score(
                __chain_vars['model'],
                __chain_vars['prev'],
                __chain_vars['target'],
                cv=folds,
                **cvargs)
        return pd.DataFrame(res, columns=['score'])
    return cross_validate_fun

def attach(module, classes=[LearnAccessor]):
    for model_class in get_models_from_module(module):
        method_name = model_class.__name__
        for cls in classes:
            setattr(cls,
                    method_name,
                    generate_model_function(model_class)
                    )
            setattr(cls,
                    'explain',
                    generate_explain_function()
                    )
            setattr(cls,
                    'cross_validate',
                    generate_cross_validate_function()
                    )

__modules_to_add = [
        sklearn.decomposition,
        sklearn.ensemble,
        sklearn.linear_model,
        sklearn.cluster,
        sklearn.preprocessing,
        sklearn.manifold
        ]


for module in __modules_to_add:
    attach(module)

