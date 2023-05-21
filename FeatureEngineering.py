import logging

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from tsfresh import extract_features, select_features
from tsfresh.utilities.dataframe_functions import roll_time_series
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import settings

from tqdm.notebook import tqdm

from Calendar import RussianBusinessCalendar


class FeatureEngineering():
    def __init__(self):
        self.series = None
        self.target = None
        self.list_of_custom_fe = dict()
        self.list_of_auto_fe = dict()
        
    def custom_fe(self, range_diff=14, range_lags=14, range_ma=14, fourier_order=14, fourier_period=365):
        # day of the week
        self.list_of_custom_fe['day_of_week'] = \
                        pd.Series(list(pd.Series(self.series.index).apply(lambda x: np.tan(x.day_of_week / 7))),
                                 index=self.series.index)
        
        # holidays
        calendar = RussianBusinessCalendar()
        holidays = [date.date() for date in calendar.get_holidays()]

        self.list_of_custom_fe['holidays'] = 1 * self.series.index.isin(holidays)
        
        # weekdays
        self.list_of_custom_fe['weekdays'] = \
                        pd.Series(list(pd.Series(self.series.index).apply(lambda x: 1 if x.day_of_week == 6 or x.day_of_week == 7 else 0)),
                                 index=self.series.index)
        
        # taxes day
        self.list_of_custom_fe['taxes_day'] = \
                            pd.Series(list(pd.Series(self.series.index).apply(lambda x: 1 if x.day == 28 else 0)),
                                      index=self.series.index)
        
        # features from dates
        for type_ in ['month', 'quarter', 
                      'year', 'weekofyear', 
                      'day_of_year', 
                      'daysinmonth', 
                      'is_leap_year',
                     'is_month_end',
                     'is_month_start',
                     'is_quarter_end',
                     'is_quarter_start',
                     'is_year_end',
                     'is_year_start']:
            self.list_of_custom_fe[type_] = \
                            pd.Series(list(1 * pd.Series(self.series.index).apply(lambda x: getattr(x, type_))),
                                     index=self.series.index)
        
        # diffs from 1 to range_diff
        for i in range(1, range_diff+1):
            self.list_of_custom_fe[f'diffs_{i}'] = self.series.diff(i).fillna(0)
            
        # lags from 1 to range_lags
        for i in range(1, range_lags+1):
            self.list_of_custom_fe[f'lag_{i}'] = self.series.shift(i).fillna(0)
        
        # rolling window stats from 2 to range_ma
        for i in range(2, range_ma+1):
            self.list_of_custom_fe[f'median_{i}'] = self.series.rolling(i, min_periods=1).median()
            self.list_of_custom_fe[f'mean_{i}'] = self.series.rolling(i, min_periods=1).mean()
            self.list_of_custom_fe[f'std_{i}'] = self.series.rolling(i, min_periods=1).std().fillna(0)
            self.list_of_custom_fe[f'max_{i}'] = self.series.rolling(i, min_periods=1).max()
            self.list_of_custom_fe[f'min_{i}'] = self.series.rolling(i, min_periods=1).min()
            
        # fourier transform
        for order in range(1, fourier_order+1):
            for func in ("sin", "cos"):
                self.list_of_custom_fe[f'fourier_{func}_{order}'] = \
                                    getattr(np, func)(2 * np.pi * self.series * order / fourier_period)
            
    def auto_fe(self, relevant_columns):
        # use tsfresh
        tmp = pd.DataFrame(list(self.series), columns=['Balance']).reset_index(drop=True).reset_index()
        tmp['index'] = tmp['index'].astype(object)
        settings_efficient = settings.EfficientFCParameters()
        extracted_features = extract_features(tmp, column_id='index', column_value="Balance",\
                                              impute_function=impute, default_fc_parameters=settings_efficient)

        extracted_features.index = self.series.index
        
        # Remove duplicates
        extracted_features = extracted_features.T.drop_duplicates().T

        if relevant_columns:
            self.list_of_auto_fe = dict(extracted_features)
        else:
            relevant_features = set()

            for label in tqdm(self.target.unique()):
                series_tmp = self.target == label
                extracted_features_filtered = select_features(extracted_features, series_tmp)
                relevant_features = relevant_features.union(set(extracted_features_filtered.columns))
                
            self.list_of_auto_fe = dict(extracted_features[list(relevant_features)])
        
        
    def get_features(self, series, target, relevant_columns=[]):
        self.series = series
        self.target = target
        self.list_of_custom_fe = dict()
        self.list_of_auto_fe = dict()
        
        self.custom_fe()
        self.auto_fe(relevant_columns)
        features = pd.DataFrame({**self.list_of_custom_fe, **self.list_of_auto_fe})
        # Remove duplicates
        features = pd.concat([pd.DataFrame(list(self.series), columns=['Balance'], 
                                           index=self.series.index), features], axis=1)
        features = features.T.drop_duplicates(keep="first").T
        if relevant_columns:
            features = features[relevant_columns]
        return features