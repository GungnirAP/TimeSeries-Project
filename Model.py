'''
Model Description:
Assumptions:
    Input time-series is bounded from 2017-01-09 to 2021-03-31.
    Metrics of model performance are measured for period from 2021-01-01 to 2021-03-30,
    simulated as it worked in inference mode.
    Rates are fixed for the whole period (training, validation, testing).
'''

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics import make_scorer, mean_absolute_error as MAE

from Preprocessing import Preprocessing
from Anomalies import AnomalyDetector
# from FeatureEngineering import FeatureEngineering
# from FeatureSelection import FeatureSelector
from ModelSelection import ModelSelector


class Machinery:
    def __init__(self, scoring):
        self.scoring = scoring

        self.Model = None
        self.features_names = None
        self.preprocessor = Preprocessing()
        self.anomaly_detector = {"income": AnomalyDetector(),
                                 "outcome": AnomalyDetector(),}
        self.feature_generator = FeatureEngineering()
        self.feature_selector = FeatureSelector(scoring=scoring, k_folds=5, k_features=50)
        self.model_selector = ModelSelector(scoring=scoring)
        
    def finetune(self, income, outcome, target):
        # Preprocessing
        income = self.preprocessor.preprocess(income)
        outcome = self.preprocessor.preprocess(outcome)
        time_series = income - outcome

        # разбить на трейн вал

        # Anomalies detection
        income = self.anomaly_detector["income"].fit(income)
        outcome = self.anomaly_detector["outcome"].fit(outcome)
        anomaly_features = pd.concat([income, outcome], axis=1).drop(columns=["val"])
        anomaly_features = anomaly_features.T.drop_duplicates().T
        anomaly_features.columns = [col + f"_{num}" for num, col in enumerate(anomaly_features.columns)] 
        raw_names = sorted(anomaly_features.columns)
        dates, weeks = 0, 0
        for name in raw_names:
            dates += 1 if name.find("date") > -1 else 0
            weeks += 1 if name.find("week") > -1 else 0
        columns = [f"irregular_date_{i}" for i in range(dates)]
        columns += [f"irregular_week_{i}" for i in range(weeks)]
        anomaly_features.columns = columns        

        # Feature Engineering
        features = self.feature_generator.get_features(time_series)
        data = pd.concat([features, anomaly_features], axis=1)
        data = data.T.drop_duplicates().T

        # Feature selection
        self.features_names = self.feature_selector.select_features(data, target)
        data = data[self.features_names]

        # Model Selection
        # на трейне отобрали faeture names (все выше на трейне)
        # сгенерили фичи заново для train+val (AD + FE)
        # data = data[self.features_names] (FS)
        # на трейне подобрали гиперы
        # на вале выбрали топ модель
        # калибруем на трейн+вал
        self.Model = self.model_selector.select_model(data, target)
        self.calibrate_model(data, target)
        return self.Model

    def calibrate_model(self, X, y):
        self.Model.fit(X, y)
        return self.Model

    def predict(self, X):
        return self.Model.predict(X)


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
    
    train_dates, test_dates = df[:'2021-01-01'].index, df['2021-01-01':'2021-03-31'].index
    income, outcome = df["Income"], df["Outcome"]
    target = (df["Income"] - df["Outcome"]).shift(-1)[:-1]

    pnl_scorer = make_scorer(pnl_score, greater_is_better=True, rates=rates)
    test_scores = []

    machine = Machinery()
    machine.finetune(income[train_dates], outcome[train_dates], target[train_dates])
    for date in test_dates:
        prediction = machine.predict(income[:date], outcome[:date], horizon=1)
        score = pnl_score(target[date], prediction)
        mae_error = MAE(target[date], prediction)
        test_scores.append((date, mae_error, score))
        machine.calibrate_model(income[:date], outcome[:date], target[:date])
    
    to_print = [f"{date.strftime('%Y-%m-%d')} {error} {score}" for date, error, score in test_scores]
    with open("test_errors.txt", "w") as f:
        f.write("\n".join(to_print))
        
    # TO DO:
    # Periodical finetune
    # Asap Finetune according to razladki points
    # Refactor Machinery finetune