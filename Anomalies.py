'''
Custom AnomalyDetector for univariate time-series.
Based on: https://github.com/denndimitrov/Timeseries/
'''

import pandas as pd
import numpy as np
from datetime import datetime

from collections import Counter
from Calendar import RussianBusinessCalendar


class AnomalyDetector(object):
    """
    Class which use CUSUM anomaly detection.
    A cumulative sum (CUSUM) chart is a type of control chart used to monitor small shifts in the process mean.
    Parameters
    ----------
    backward_window_size : integer, optional, default 30
        The window size of timeseries for estimate stats (like train)
    forward_window_size : integer, optional, default 14
        The window size of timeseries for compare with backward_window_size (like test)
    threshold : float, optional, default 5.0
        The maximum(minimum, with opposite sign) value of cumulative changes
    drift : float, optional, default 1.0
        The permissible deviation of timeseries from the mean
    Attributes
    ----------
    anomalies : pd.DataFrame with initial time-series index containing features
                    val : observed values
                    irregular_date_{i} : binary values (1 - irregular date, 0 - regular date)
                    irregular_week_{i} : binary values (1 - irregular week, 0 - regular week)
    """

    def __init__(self, backward_window_size=28, forward_window_size=7, threshold=3.5, drift=1.0):
        self.backward_window_size = backward_window_size
        self.forward_window_size = forward_window_size
        self.threshold = threshold
        self.drift = drift

        calendar = RussianBusinessCalendar()
        self.holidays = [date.date() for date in calendar.get_holidays()]
        self.holidays.append(datetime.strptime('10-11-2017', "%d-%m-%Y").date())
        self.holidays.append(datetime.strptime('10-10-2018', "%d-%m-%Y").date())

        self.irregular_dates = None
        # Empirically defined that many outliers occur in 52 and 13 weeks of year:
        self.irregular_weeks = [13, 52]

    def one_pass(self, train_zone, prediction_zone, threshold=None, drift=None):
        """
        Detect anomaly in one pass
        Parameters
        ----------
        train_zone : pandas.Series or pandas.DataFrame
            Train sample to calculate statistics of timeseries
        prediction_zone : pandas.Series or pandas.DataFrame
            Test sample to find anomaly variables
        threshold : float, optional, default 5.0
            See parameter in ``threshold`` in :class:`AnomalyDetector`:func:`__init__`
        drift : float, optional, default 1.0
            See parameter in ``drift`` in :class:`AnomalyDetector`:func:`__init__``
        Returns
        -------
        is_fault : binary numpy array, shape = [len(prediction_zone)]
            1 - anomaly, 0 - nonanomaly
        """

        if not threshold:
            threshold = self.threshold
        if not drift:
            drift = self.drift

        current_std = np.nanstd(train_zone, ddof=1)
        current_mean = np.nanmean(train_zone)
        drift = drift * current_std
        threshold = threshold * current_std

        x = prediction_zone.astype('float64')
        gp, gn = np.zeros(x.size), np.zeros(x.size)

        for i in range(1, x.size):
            gp[i] = max(gp[i - 1] + x[i] - current_mean - drift, 0)
            gn[i] = min(gn[i - 1] + x[i] - current_mean + drift, 0)

        is_fault = np.logical_or(gp > threshold, gn < -threshold)
        return is_fault

    def detect(self, time_series, excluded_points=None):
        """
        Detect anomaly in rolling window (=forward_window_size)
        Parameters
        ----------
        time_series : pandas.Series
            Target timeseries
        threshold : float, optional, default 5.0
            See parameter in ``threshold`` in :class:`AnomalyDetector`:func:`__init__`
        drift : float, optional, default 1.0
            See parameter in ``drift`` in :class:`AnomalyDetector`:func:`__init__``
        excluded_points : pandas.Series.index
            Acquainted anomaly events. They will be removed from timeseries before anomaly detection
        Returns
        -------
        anomalies : pandas.Series, shape = [len(time_series)]
            Labeled timeseries with anomaly, where 1 - anomaly, 0 - nonanomaly
        """
        if excluded_points is not None:
            time_series[time_series.index.isin(excluded_points)] = np.nan
        
        ts_values = time_series.values
        ts_index = time_series.index

        detection_series = np.zeros(len(ts_values)).astype('int32')

        for ini_index in range(len(ts_values) - (self.backward_window_size + self.forward_window_size)):
            sep_index = ini_index + self.backward_window_size
            end_index = sep_index + self.forward_window_size
            faults_indexes = self.one_pass(ts_values[ini_index:sep_index],
                                           ts_values[sep_index:end_index])
            detection_series[sep_index:end_index][faults_indexes] = 1
        anomalies = pd.Series(detection_series, index=ts_index)

        return anomalies != 0

    def generate_calendar_features(self, time_series):
        data = pd.DataFrame(time_series, columns=["val"])
        data['date'] = data.index.date
        data['week_day'] = data['date'].apply(lambda x: x.weekday())
        data['month_day'] = data['date'].apply(lambda x: x.day)
        data['holiday'] = data['date'].apply(lambda x: x in self.holidays)
        data['week_of_year'] = [date.weekofyear for date in data.index]
        return data

    def fit(self, time_series):
        data = self.generate_calendar_features(time_series)
        excluded_dates = data[(data.week_day.isin([5, 6]) | data['holiday'])].index
        first_outliers = self.detect(data.val.copy(), excluded_points=excluded_dates)

        data["is_anomaly"] = (first_outliers != 0).astype(int)
        
        self.irregular_dates = AnomalyDetector.filter_irregular_dates(first_outliers, freq="days")
        output = self.generate_irregular_features(time_series)
        return output

    def generate_irregular_features(self, time_series):
        data = self.generate_calendar_features(time_series)
        for num, day in enumerate(self.irregular_dates):
            data[f"irregular_date_{num}"] = data["month_day"].apply(lambda x: x == day).astype(int)
        for num, week in enumerate(self.irregular_weeks):
            data[f"irregular_week_{num}"] = data["week_of_year"].apply(lambda x: x == week).astype(int)
        data.drop(columns=["date", "week_day", "month_day", "holiday", "week_of_year"], inplace=True)
        return data


    @staticmethod
    def filter_irregular_dates(is_outlier, th=0.5, freq="days"):
        if freq == "days":
            anomaly_idx = [i.day for i in is_outlier[is_outlier].index]
        elif freq == "week_days":
            anomaly_idx = [i.weekday() for i in is_outlier[is_outlier].index]
        else:
            pass

        count_anomalies_by_idx = sorted(Counter(anomaly_idx).items(), key=lambda x: x[1], reverse=True)
        idx = [x[0] for x in count_anomalies_by_idx]
        anomaly_count = [x[1] for x in count_anomalies_by_idx]
        
        cum_anomalies = np.cumsum(anomaly_count)
        temp = (cum_anomalies / sum(anomaly_count)) <= th

        return idx[:temp[temp].shape[0]]