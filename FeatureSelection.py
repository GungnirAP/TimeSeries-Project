from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn.feature_selection import VarianceThreshold, SelectFromModel
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from pyitlib import discrete_random_variable as drv
from tqdm.notebook import tqdm

import pandas as pd
import numpy as np

class FeatureSubsetGenerator():
    def __init__(self, features, target, scoring='neg_mean_absolute_error', k_features=50):
        self.features = features
        self.target = target
        self.k_features = k_features
        self.scoring = scoring
        
    # Оберточный метод – SequentialFeatureSelector
    def wrapper_method(self):
        feature_selector = SequentialFeatureSelector(Lasso(),
           k_features=self.k_features,
           forward=True,
           verbose=2,
           scoring=self.scoring,
           cv=4)
        features = feature_selector.fit(np.array(self.features.fillna(0)), self.target)
        return self.features.columns[list(features.k_feature_idx_)].values
    
    # Фильтр – корреляция
    def correlation_method(self):
        corr = dict()
        for i in self.features.columns:
            corr[i] = [self.target.corr(self.features[i])]
        self.correlation_features = \
                self.features[pd.DataFrame(corr).T.sort_values(0, 
                                                               key=abs, 
                                                               ascending=False).iloc[:self.k_features].index]
        return self.correlation_features.columns.values

    # Фильтр – Mutual Information
    def filtered_method(self):
        # Remove quasi-constant features
        filter_ = VarianceThreshold(threshold=0.01)
        filter_.fit(self.features)
        self.filter_features = filter_.transform(self.features)
        self.filter_features =  pd.DataFrame(self.filter_features, index=self.features.index,
                                            columns=self.features.columns[filter_.get_support()])
        
        # mutual info conditional
        chosen_columns = dict()
        chosen_columns['const'] = pd.Series([1] * len(self.target), index=self.features.index)
        for i in tqdm(range(self.k_features)):
            mi_cumsum = dict()
            for X in self.filter_features.columns:
                mi_cumsum[X] = 0
                for Z in chosen_columns:
                    mi_cumsum[X] += drv.information_mutual_conditional(self.features[X].values.flatten(), 
                                               self.target.values, 
                                               chosen_columns[Z].values.flatten())
            chosen_feature = sorted(mi_cumsum.items(), key=lambda x:x[1])[-1][0]
            chosen_columns[chosen_feature] = self.features[chosen_feature]
            self.filter_features = self.filter_features.drop(columns=chosen_feature)

        return np.array(chosen_columns.keys())
    
    # Встроенные – Lasso, Ridge, Elastic Net
    def embedded_method(self):
        sel_lasso = SelectFromModel(Lasso(alpha=0.001, random_state=10))
        sel_lasso.fit(self.features, self.target)
        
        sel_ridge = SelectFromModel(Ridge(alpha=0.001, random_state=10))
        sel_ridge.fit(self.features, self.target)
        
        sel_elastic = SelectFromModel(ElasticNet(alpha=0.001, random_state=10))
        sel_elastic.fit(self.features, self.target)
        
        return self.features.columns[sel_lasso.get_support()].values,\
               self.features.columns[sel_ridge.get_support()].values,\
               self.features.columns[sel_elastic.get_support()].values