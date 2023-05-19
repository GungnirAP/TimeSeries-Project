import logging

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from tsfresh import extract_features, select_features
from tsfresh.utilities.dataframe_functions import roll_time_series
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import settings

from tqdm.notebook import tqdm


class FeatureEngineering():
    def __init__(self, series):
        self.series = series
        self.list_of_custom_fe = dict()
        self.list_of_auto_fe = dict()
        
    def custom_fe(self, range_diff=14, range_lags=14, range_ma=14, fourier_order=14, fourier_period=365):
        # day of the week
        self.list_of_custom_fe['day_of_week'] = \
                        pd.Series(list(pd.Series(self.series.index).apply(lambda x: np.tan(x.day_of_week / 7))),
                                 index=self.series.index)
        
        # weekdays
        self.list_of_custom_fe['weekdays'] = \
                        pd.Series(list(pd.Series(self.series.index).apply(lambda x: 1 if x.day_of_week == 6 or x.day_of_week == 7 else 0)),
                                 index=self.series.index)
        # month
        self.list_of_custom_fe['month'] = \
                        pd.Series(list(pd.Series(self.series.index).apply(lambda x: x.month)),
                                 index=self.series.index)
        
        # quarter
        self.list_of_custom_fe['quarter'] = \
                        pd.Series(list(pd.Series(self.series.index).apply(lambda x: x.quarter)),
                                 index=self.series.index)
        
        # taxes day
        self.list_of_custom_fe['taxes_day'] = \
                            pd.Series(list(pd.Series(self.series.index).apply(lambda x: 1 if x.day == 28 else 0)),
                                      index=self.series.index)
        
        # diffs from 1 to range_diff
        for i in range(1, range_diff+1):
            self.list_of_custom_fe[f'diffs_{i}'] = self.series.diff(i).fillna(0)
            
        # lags from 1 to range_lags
        for i in range(1, range_lags+1):
            self.list_of_custom_fe[f'lag_{i}'] = self.series.shift(i).fillna(0)
        
        # rolling window stats from 2 to range_ma
        for i in range(2, range_ma+1):
            # MA
            self.list_of_custom_fe[f'ma_{i}'] = self.series.rolling(i, min_periods=1).mean()
            # median
            self.list_of_custom_fe[f'ma_{i}'] = self.series.rolling(i, min_periods=1).median()
            # std
            self.list_of_custom_fe[f'ma_{i}'] = self.series.rolling(i, min_periods=1).std()
            # max
            self.list_of_custom_fe[f'ma_{i}'] = self.series.rolling(i, min_periods=1).max()
            # min
            self.list_of_custom_fe[f'ma_{i}'] = self.series.rolling(i, min_periods=1).min()
            
        # fourier transform
        for order in range(1, fourier_order+1):
            for func in ("sin", "cos"):
                self.list_of_custom_fe[f'fourier_{func}_{order}'] = \
                                    getattr(np, func)(2 * np.pi * self.series * order / fourier_period)
            
    def auto_fe(self):
        # use tsfresh
        tmp = pd.DataFrame(self.series, columns=['Balance']).reset_index(drop=True).reset_index()
        tmp['index'] = tmp['index'].astype(object)
        settings_efficient = settings.EfficientFCParameters()
        extracted_features = extract_features(tmp, column_id='index', column_value="Balance",\
                                              impute_function=impute, default_fc_parameters=settings_efficient)

        extracted_features.index = self.series.index
        
        # Remove duplicates
        extracted_features = extracted_features.T.drop_duplicates().T
        relevant_features = set()
        
        # match tomorrow balance and current data
        target = self.series.shift(-1)[:-1]
        extracted_features = extracted_features.iloc[:-1]

        for label in tqdm(target.unique()):
            series_tmp = target == label
            extracted_features_filtered = select_features(extracted_features, series_tmp)
            relevant_features = relevant_features.union(set(extracted_features_filtered.columns))
            

            
        self.list_of_auto_fe = dict(extracted_features[list(relevant_features)])
        
        
    def get_features(self):
        self.custom_fe()
        self.auto_fe()
        features = pd.DataFrame({**self.list_of_custom_fe, **self.list_of_auto_fe}).iloc[:-1]
        # Remove duplicates
        features = features.T.drop_duplicates().T
        return features