import logging

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class Preprocessing():
    def __init__(self):
        pass
    
    def preprocess(self, series, index_name='Date'):
        # all dates are unique
        counter = series.groupby(index_name).count().sort_values()
        if len(counter[counter > 1]) > 0:
            for date in counter[counter > 1].index:
                logging.warning(f"Duplicate date: {date.date()}")
            logging.warning("Preprocessing is not completed.")
            return
        
        # all dates are mentioned
        period = pd.date_range(start=series.index[0], end=series.index[-1])
        difference = set(period).difference(series.index)
        if difference:
            logging.info(f"Number of dates to be filled: {len(difference)}")
            
        series = pd.concat([series, pd.Series(data=[np.nan] * len(difference), index=list(difference))])
        series = series.sort_index()
        
        # fill nans - linear interpolate
        series = series.interpolate()

        return series