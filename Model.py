'''
Model Description:
Assumptions:
    Input time-series is bounded from 2017-01-09 to 2021-03-31.
    Metrics of model performance are measured for period from 2021-01-01 to 2021-03-21
    if it is worked in inference mode.
    Rates are fixed for the whole period (training, validation, testing).
'''

import pandas as pd
from datetime import datetime
# from sklearn.metrics import mean_absolute_error as MAE

from Preprocessing import Preprocessing
from Anomalies import AnomalyDetector
from FeatureEngineering import FeatureEngineering
from FeatureSelection import FeatureSelector
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
        self.model_selector = ModelSelector()
        
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
        # на трейне отобрали faeture names (все ввыше на трейне)
        # сгенерили заново для train+val 
        # data = data[self.features_names]
        # на трейне подобрали гиперы
        # на вале выбрали топ модель
        # калибруем на трейн+вал
        self.Model = self.model_selector.select_model(data, target)
        self.calibrate_model(data, target)
        return self.Model

    def calibrate_model(self, data, target):
        self.Model.fit(data, target)
        return self.Model

    def predict(self, X):
        pass
        # self.Model.predict(X)

if __name__ == '__main__':
    rates = pd.read_csv("./data/input_rates.csv", index_col=0).values
    rates = dict(rates)

    date_parser = lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S.%f")
    df = pd.read_excel('data/Project 2_2023.xlsx', sheet_name='Data', 
                       parse_dates=['Date'], date_parser=date_parser)
    df = df.set_index('Date')
    df.index.name = 'Date'

    test_dates = df['2021-01-01':].index
    scores = []

    # scoring
    # train_test_split
    # machine = Machinery()
    # machine.finetune(train)
    # for date in test_dates:
        # machine.predict(test[:date])
        # score = согласно скорингу
        # scores.append((date, score))
        # machine.calibrate_model(new_train)