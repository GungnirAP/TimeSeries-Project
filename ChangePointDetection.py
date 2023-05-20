# based on
# https://www.sarem-seitz.com/probabilistic-cusum-for-change-point-detection/

import torch
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

from typing import Tuple
from Calendar import RussianBusinessCalendar


class CusumMeanDetector():
    def __init__(self, t_warmup = 30, p_limit = 0.1) -> None:
        self._t_warmup = t_warmup
        self._p_limit = p_limit
        
        self._reset()

    def predict_next(self, y: torch.tensor) -> Tuple[float,bool]:
        self._update_data(y)

        if self.current_t == self._t_warmup:
            self._init_params()
        
        if self.current_t >= self._t_warmup:
            prob, is_changepoint = self._check_for_changepoint()
            if is_changepoint:
                self._reset()

            return (1-prob), is_changepoint
        
        else:
            return 0, False
            
    def _reset(self) -> None:
        self.current_t = torch.zeros(1)
                
        self.current_obs = []
        
        self.current_mean = None
        self.current_std = None
    
    def _update_data(self, y: torch.tensor) -> None:
        self.current_t += 1
        self.current_obs.append(y.reshape(1))

        
    def _init_params(self) -> None:
        self.current_mean = torch.mean(torch.concat(self.current_obs))
        self.current_std = torch.std(torch.concat(self.current_obs))
             
    def _check_for_changepoint(self) -> Tuple[float,bool]:
        standardized_sum = torch.sum(torch.concat(self.current_obs) - self.current_mean)/(self.current_std * self.current_t**0.5)
        prob = float(self._get_prob(standardized_sum).detach().numpy())
        
        return prob, prob < self._p_limit
    
    def _get_prob(self, y: torch.tensor) -> bool:
        p = torch.distributions.normal.Normal(0,1).cdf(torch.abs(y))
        prob = 2*(1 - p)
        
        return prob
    
class ChangePointDetector():
    def __init__(self):
        self.initial_dates = None
        self.date_no_holidays = None
        self.series = None
        self.series_no_holidays = None
        self.detector = None
        self.outs = None
        self.cps = None
        self.cps_dates = None
        self.probs = None
        
        calendar = RussianBusinessCalendar()
        self.holidays = [date.date() for date in calendar.get_holidays()]

        
    def delete_holidays(self, series):
        delete_days = np.where(series.index.isin(self.holidays))[0]
        no_holidays = series.drop(series.index[delete_days])
        weekends = pd.Series(no_holidays.index).apply(lambda x: True if x.dayofweek in [5,6] else False)
        
        self.series_no_holidays = no_holidays.drop(no_holidays.index[weekends])
        self.date_no_holidays = pd.Series(self.series_no_holidays.index)
        self.series_no_holidays = torch.tensor(self.series_no_holidays.values)
    
    def detect_changepoint(self, series):
        self.initial_dates = pd.Series(series.index)
        self.series = torch.tensor(series.values)
        self.delete_holidays(series)
        
        self.detector = CusumMeanDetector()
        self.outs = [self.detector.predict_next(
            self.series_no_holidays[i]) for i in range(len(self.series_no_holidays))]
        
        self.cps = np.where(list(map(lambda x: x[1], self.outs)))[0]
        self.probs = np.array(list(map(lambda x: x[0], self.outs)))
        
        self.cps_dates = self.date_no_holidays[self.cps]
        self.cps = np.where(self.initial_dates.isin(self.cps_dates))[0]
        return max(self.cps_dates)
    
    def plot_changepoints(self):
        X, Y = np.meshgrid(np.arange(len(self.series)),np.linspace(min(self.series), max(self.series)))

        plt.figure(figsize=(18,9))
        plt.plot(np.arange(len(self.series)),self.series.detach().numpy(),lw=0.75,label="Data",color="blue")

        plt.axvline(self.cps[0], color="red", linestyle="dashed",label="Detected changepoints",lw=2)
        [plt.axvline(cp, color="red", linestyle="dashed",lw=2) for cp in self.cps[1:]]

        plt.legend()
        
    def return_minimum_period(self):
        to_count = pd.Series([self.initial_dates[0]-timedelta(days=1), *self.cps_dates])
        periods = []
        for i in range(len(to_count) - 1):
            periods.append(to_count.iloc[i + 1] - to_count.iloc[i])
        return min(periods).days