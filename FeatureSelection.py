from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn.feature_selection import VarianceThreshold, SelectFromModel
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from pyitlib import discrete_random_variable as drv

from tqdm.notebook import tqdm

import warnings
warnings.filterwarnings("ignore")

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

        return np.array([*chosen_columns.keys()])
    
    # Встроенные – Lasso, Ridge, Elastic Net
    def embedded_method_lasso(self):
        sel_lasso = SelectFromModel(Lasso(alpha=0.001, random_state=10))
        sel_lasso.fit(self.features, self.target)
        return self.features.columns[sel_lasso.get_support()].values
        
    def embedded_method_ridge(self):
        sel_ridge = SelectFromModel(Ridge(alpha=0.001, random_state=10))
        sel_ridge.fit(self.features, self.target)
        return self.features.columns[sel_ridge.get_support()].values
        
    def embedded_method_elastic(self):
        sel_elastic = SelectFromModel(ElasticNet(alpha=0.001, random_state=10))
        sel_elastic.fit(self.features, self.target)
        return self.features.columns[sel_elastic.get_support()].values
    
class FeatureSelector():
    def __init__(self, k_folds=5, scoring='neg_mean_absolute_error', k_features=50):
        self.scoring = scoring
        self.k_features = k_features
        self.k_folds = k_folds
    
    def select_features(self, features, target, use_correlation=False): # np array of features names
        step = len(features) / self.k_folds
        list_of_samples = []
        list_of_targets = []
        for i in range(self.k_folds):
            list_of_samples.append(features[int(i*step) : int((i+1)*step)])
            list_of_targets.append(target[int(i*step) : int((i+1)*step)])

        methods = ['embedded_method_lasso', 
                   'embedded_method_ridge',
                   'embedded_method_elastic',
                   'filtered_method']
        if use_correlation:
            methods.append('correlation_method')
        feature_names = pd.Series(features.columns.values)

        scores_by_method = dict()

        # Nogueira method
        for method in tqdm(methods):
            chosen_features = []
            k_avg = 0

            for i in range(self.k_folds):
                fsg = FeatureSubsetGenerator(list_of_samples[i], list_of_targets[i], k_features=self.k_features)
                class_method = getattr(fsg, method)
                chosen_features.append(class_method())
                k_avg += len(chosen_features[-1])

            k_avg /= self.k_folds

            frequency = []
            for set_ in chosen_features:
                frequency.append(np.array(1 * feature_names.isin(set_)))
            frequency = np.array(frequency)

            p_t = frequency.sum(axis=0) / self.k_folds
            s_t2 = (p_t * (1 - p_t) * self.k_folds / (self.k_folds - 1))
            score = 1 - (s_t2.sum(axis=0) / len(feature_names)) / \
                                    (k_avg / len(feature_names) * (1 - k_avg / len(feature_names)))
            scores_by_method[method] = score

        best_method = sorted(scores_by_method.items(), key=lambda x:x[1])[-1][0]
        fsg = FeatureSubsetGenerator(features, target, k_features=self.k_features)
        class_method = getattr(fsg, best_method)
        return class_method()
