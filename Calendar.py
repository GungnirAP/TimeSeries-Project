import pandas as pd
import datetime
from pandas.tseries.holiday import AbstractHolidayCalendar, Holiday

class RussianBusinessCalendar(AbstractHolidayCalendar):
    def __init__(self):   
        self.start_date = datetime.datetime(2017, 1, 1)
        self.end_date = datetime.datetime(2023, 6, 11)
        russian_calendar = pd.read_csv('./data/holidays.csv', index_col=0)
        self.russian_calendar = {"holidays": pd.to_datetime(russian_calendar['date'])}
        
    def get_holidays(self):
        rules = [Holiday(name='Russian Day Off', year=d.year, month=d.month, day=d.day) 
                 for d in self.russian_calendar['holidays']
                ]
        return rules