'''
Model Description:
    Class Machinery supports automatic refit and 1-day ahead forecasts 
    for daily saldos of income and outcome. The main part contains the 
    script of the experiment under the following assumptions.
Assumptions:
    Model predicts for 1-day ahead for the history of income and outcome.
    Input time-series is bounded from 2017-01-09 to 2021-03-31.
    Metrics of model performance are measured for period from 2021-01-01 to 2021-03-30,
    simulated as it worked in inference mode.
    Rates are fixed for the whole period (training, validation, testing).
'''

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics import make_scorer, mean_absolute_error as MAE

from ChangePointDetection import ChangePointDetector


from Calendar import RussianBusinessCalendar
from Preprocessing import Preprocessing
from Anomalies import AnomalyDetector
from FeatureEngineering import FeatureEngineering
from FeatureSelection import FeatureSelector
from ModelSelection import ModelSelector

class Machinery:
    def __init__(self, score, scorer, finetune_every=49, k_features=50, n_jobs=None):
        self.score = score
        self.scorer = scorer
        self.finetune_every = finetune_every
        self.finetune_count = finetune_every
        self.n_jobs = n_jobs

        self.Model = None
        self.features_names = None
        self.preprocessor = Preprocessing()
        self.anomaly_detector = {"income": AnomalyDetector(),
                                 "outcome": AnomalyDetector(),}
        self.feature_generator = FeatureEngineering()
        self.feature_selector = FeatureSelector(scoring=scorer, k_folds=5, k_features=k_features)
        self.model_selector = ModelSelector(scoring=scorer, pnl_score=score, n_jobs=self.n_jobs)

        calendar = RussianBusinessCalendar()
        self.holidays = [date.date() for date in calendar.get_holidays()]
        self.holidays.append(datetime.strptime('10-11-2017', "%d-%m-%Y").date())
        self.holidays.append(datetime.strptime('10-10-2018', "%d-%m-%Y").date())

        self.all_irregular_dates = None
        self.all_irregular_weeks = None
        

    def generate_irregular_features(self, time_series, irregular_dates, irregular_weeks):
        data = pd.DataFrame(time_series, columns=["val"])
        data['date'] = data.index.date
        data['week_day'] = data['date'].apply(lambda x: x.weekday())
        data['month_day'] = data['date'].apply(lambda x: x.day)
        data['holiday'] = data['date'].apply(lambda x: x in self.holidays)
        data['week_of_year'] = [date.weekofyear for date in data.index]
        for day in irregular_dates:
            data[f"irregular_date_{day}"] = data["month_day"].apply(lambda x: x == day).astype(int)
        for week in irregular_weeks:
            data[f"irregular_week_{week}"] = data["week_of_year"].apply(lambda x: x == week).astype(int)
        data.drop(columns=["val", "date", "week_day", "month_day", "holiday", "week_of_year"], inplace=True)
        return data

    def finetune(self, income, outcome, target, val_size=49):
        if self.finetune_count >= self.finetune_every:
            self.finetune_count = 0

            # Preprocessing
            income = self.preprocessor.preprocess(income)
            outcome = self.preprocessor.preprocess(outcome)
            time_series = income - outcome

            train_index = income[:-val_size].index
            val_index = income[-val_size:].index

            # Anomalies detection
            self.anomaly_detector["income"].fit(income[train_index])
            self.anomaly_detector["outcome"].fit(outcome[train_index])
            all_irregular_dates = set(self.anomaly_detector["income"].irregular_dates)
            all_irregular_dates = all_irregular_dates.union(set(self.anomaly_detector["outcome"].irregular_dates))
            all_irregular_weeks = self.anomaly_detector["income"].irregular_weeks
            anomaly_features = self.generate_irregular_features(income[train_index], all_irregular_dates, all_irregular_weeks)     
            self.all_irregular_dates = all_irregular_dates
            self.all_irregular_weeks = all_irregular_weeks

            # Feature Engineering
            features = self.feature_generator.get_features(time_series[train_index], target[train_index])
            train_data = pd.concat([features, anomaly_features], axis=1)
            train_data = train_data.T.drop_duplicates().T

            # Feature selection
            self.features_names = self.feature_selector.select_features(train_data, target[train_index])
            if "Balance" not in self.features_names:
                self.features_names = np.append(self.features_names, "Balance")
            
            # Model Selection
            anomaly_features = self.generate_irregular_features(income, all_irregular_dates, all_irregular_weeks)     
            features = self.feature_generator.get_features(time_series, target)
            val_data = pd.concat([features, anomaly_features], axis=1)
            val_data = val_data.T.drop_duplicates().T
            val_data = val_data[self.features_names]
            self.Model = self.model_selector.select_model(val_data, target, train_index, val_index)
            self.__calibrate_model(val_data, target)

        return self.Model

    def __calibrate_model(self, val_data, target):
        self.Model.fit(val_data, target)
        return self.Model

    def calibrate_model(self, income, outcome, target):
        # Preprocessing
        income = self.preprocessor.preprocess(income)
        outcome = self.preprocessor.preprocess(outcome)
        time_series = income - outcome

        anomaly_features = self.generate_irregular_features(time_series, 
                                                            self.all_irregular_dates, 
                                                            self.all_irregular_weeks) 
        new_columns = list(set(self.features_names).difference(set(anomaly_features.columns)))
        features = self.feature_generator.get_features(time_series, target=[], 
                                                       relevant_columns=new_columns)    
        val_data = pd.concat([features, anomaly_features], axis=1)
        val_data = val_data.T.drop_duplicates().T
        val_data = val_data[self.features_names]
        self.Model.fit(val_data, target)
        return self.Model

    def predict(self, income, outcome): 
        # Preprocessing
        income = self.preprocessor.preprocess(income)
        outcome = self.preprocessor.preprocess(outcome)
        time_series = income - outcome

        anomaly_features = self.generate_irregular_features(time_series, 
                                                            self.all_irregular_dates, 
                                                            self.all_irregular_weeks) 
        new_columns = list(set(self.features_names).difference(set(anomaly_features.columns)))
        features = self.feature_generator.get_features(time_series, target=[], 
                                                       relevant_columns=new_columns)    
        val_data = pd.concat([features, anomaly_features], axis=1)
        val_data = val_data.T.drop_duplicates().T
        val_data = val_data[self.features_names]
        preds = self.Model.predict(val_data.iloc[-1])
        date = preds.index[0].date()
        if (date.weekday() in [5, 6]) or (date in self.holidays):
            preds[0] = 0
        self.finetune_count += 1
        return preds


def pnl_score(y_true, y_predict, 
              rates={"key_rate" : 7.25, 
                     "deposit_rate" : -0.9, 
                     "credit_rate" : 1.0, 
                     "profit_rate" : 0.5}):
    y_true = np.array(y_true)
    y_predict = np.array(y_predict)
    diffs = y_predict - y_true
    multiplier = rates["profit_rate"] - (diffs > 0) * rates["credit_rate"] - (diffs < 0) * rates["deposit_rate"]
    return (diffs * multiplier / 365).mean()


if __name__ == '__main__':
    rates = pd.read_csv("./data/input_rates.csv", index_col=0).values
    rates = dict(rates)

    date_parser = lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S.%f")
    df = pd.read_excel('data/Project 2_2023.xlsx', sheet_name='Data', 
                       parse_dates=['Date'], date_parser=date_parser)
    df = df.set_index('Date')
    df.index.name = 'Date'
    
    train_dates, test_dates = df[:'2020-12-31'].index, df['2021-01-01':'2021-03-31'].index[:-1]
    income, outcome = df["Income"], df["Outcome"]
    target = (df["Income"] - df["Outcome"]).shift(-1)[:-1]

    pnl_scorer = make_scorer(pnl_score, greater_is_better=True, rates=rates)

    change_point_detector = ChangePointDetector()

    machine = Machinery(score=pnl_score, scorer=pnl_scorer, n_jobs=None)
    machine.finetune(income[train_dates], outcome[train_dates], target[train_dates])

    all_preds = []
    all_targets = []
    for date in test_dates:
        force_finetune = False
        for series in [income, outcome]:
            last_chp_date = change_point_detector.detect_changepoint(series[:date])
            if last_chp_date:
                if (date - last_chp_date).days < machine.finetune_every:
                    force_finetune = True
                    break
        if force_finetune:
            machine.finetune_count = machine.finetune_every
            machine.finetune(income[:date][:-1], outcome[:date][:-1], target[:date][:-1])

        prediction = machine.predict(income[:date], outcome[:date])
        all_preds.append(prediction)
        all_targets.append(target[date])
        machine.finetune(income[:date], outcome[:date], target[:date])
        machine.calibrate_model(income[:date], outcome[:date], target[:date])
    
    output = pd.DataFrame([np.array(all_preds).T[0],  np.array(all_targets)]).T
    output.columns = ["prediction", "fact"]
    output.index = test_dates
    output.to_csv("experiment_result.csv")
