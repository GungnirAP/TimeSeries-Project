import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

import pmdarima as pm
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn import svm

from tqdm import tqdm

class Model:
    def __init__(self, 
                 model_type="SARIMA", 
                 scoring=None,
                 hyparameters=None,
                 main_column="Balance"):
        self.main_column = main_column
        self.model_type = model_type
        self.hyperparameters = hyparameters
        self.scoring = scoring

        self.model = None
        self.pipeline = None

    def predict(self, X, horizon=1):
        assert horizon == 1
        if self.model_type == "SARIMA":
            preds = self.model.predict(n_periods=horizon)
        elif self.model_type == "SARIMAX":
            X = X.drop(columns=self.main_column)
            X = pd.DataFrame(X).T
            preds = self.model.predict(n_periods=horizon, X=X)
        elif self.model_type in ["LinearRegression", "Lasso", "Ridge", "ElasticNet", "SVM", "GradientBoostingRegressor"]:
            X = pd.DataFrame(X).T
            index = [X.index[-1]]
            if self.model_type != "GradientBoostingRegressor":
                X = self.pipeline.transform(X)
            preds = pd.Series(self.model.predict(X), index=index)
        return preds
    
    def fit(self, X, y):
        series = X[self.main_column]
        if self.model_type == "SARIMA":
            self.model = pm.ARIMA(order=self.hyperparameters["order"])
            self.model.fit(series)
        elif self.model_type == "SARIMAX":
            self.model = pm.ARIMA(order=self.hyperparameters["order"])
            self.model.fit(series, X.drop(columns=self.main_column))
        elif self.model_type == "LinearRegression":
            self.model = LinearRegression(**self.hyperparameters)
            self.pipeline = StandardScaler()
            self.pipeline.fit(X)
            X =self.pipeline.transform(X)
            self.model.fit(X, y)
        elif self.model_type == "Lasso":
            self.model = Lasso(**self.hyperparameters)
            self.pipeline = StandardScaler()
            self.pipeline.fit(X)
            X =self.pipeline.transform(X)
            self.model.fit(X, y)
        elif self.model_type == "Ridge":
            self.model = Ridge(**self.hyperparameters)
            self.pipeline = StandardScaler()
            self.pipeline.fit(X)
            X =self.pipeline.transform(X)
            self.model.fit(X, y)
        elif self.model_type == "ElasticNet":
            self.model = ElasticNet(**self.hyperparameters)
            self.pipeline = StandardScaler()
            self.pipeline.fit(X)
            X =self.pipeline.transform(X)
            self.model.fit(X, y)
        elif self.model_type == "SVM":
            self.model = svm.SVR(**self.hyperparameters)
            self.pipeline = StandardScaler()
            self.pipeline.fit(X)
            X =self.pipeline.transform(X)
            self.model.fit(X, y)
        elif self.model_type == "GradientBoostingRegressor":
            self.model = GradientBoostingRegressor(**self.hyperparameters)
            self.model.fit(X, y)
            
        return self.model

    def optimize_hyperparameters(self, X, y):
        series = X[self.main_column]
        if self.model_type == "SARIMA":
            self.model = pm.auto_arima(y=series, d=1, max_p=10, max_q=10, 
                                       seasonal=True, error_action='ignore', 
                                       suppress_warnings=True, stepwise=True)
            self.hyperparameters = {"order" : self.model.get_params()["order"]}
        elif self.model_type == "SARIMAX":
            self.model = pm.auto_arima(y=series, X=X.drop(columns=self.main_column), 
                                       d=1, max_p=10, max_q=10, 
                                       seasonal=True, error_action='ignore', 
                                       suppress_warnings=True, stepwise=True)
            self.hyperparameters = {"order" : self.model.get_params()["order"]}
        elif self.model_type == "LinearRegression":
            self.model = LinearRegression()
            parameters = {"fit_intercept":[False, True]}
            grid_GBR = GridSearchCV(estimator=self.model, param_grid = parameters, cv = 5, n_jobs=-1, scoring=self.scoring)
            self.pipeline = StandardScaler()
            self.pipeline.fit(X)
            X =self.pipeline.transform(X)
            grid_GBR.fit(X, y)
            self.hyperparameters = grid_GBR.best_params_
        elif self.model_type == "Lasso":
            self.model = Lasso()
            parameters = {"alpha":np.logspace(-5, 1, 19),
                          "fit_intercept":[False, True]}
            grid_GBR = GridSearchCV(estimator=self.model, param_grid = parameters, cv = 5, n_jobs=-1, scoring=self.scoring)
            self.pipeline = StandardScaler()
            self.pipeline.fit(X)
            X =self.pipeline.transform(X)
            grid_GBR.fit(X, y)
            self.hyperparameters = grid_GBR.best_params_
        elif self.model_type == "Ridge":
            self.model = Ridge()
            parameters = {"alpha":np.logspace(-5, 2, 23),
                         "fit_intercept":[False, True],
                         "solver":["auto", "svd", "cholesky"]}
            grid_GBR = GridSearchCV(estimator=self.model, param_grid = parameters, cv = 5, n_jobs=-1, scoring=self.scoring)
            self.pipeline = StandardScaler()
            self.pipeline.fit(X)
            X =self.pipeline.transform(X)
            grid_GBR.fit(X, y)
            self.hyperparameters = grid_GBR.best_params_
        elif self.model_type == "ElasticNet":
            self.model = ElasticNet()
            parameters = {"alpha":np.logspace(-5, 2, 8),
                         "l1_ratio":np.linspace(0, 1, 11),
                         "fit_intercept":[False, True]}
            grid_GBR = GridSearchCV(estimator=self.model, param_grid = parameters, cv = 5, n_jobs=-1, scoring=self.scoring)
            self.pipeline = StandardScaler()
            self.pipeline.fit(X)
            X =self.pipeline.transform(X)
            grid_GBR.fit(X, y)
            self.hyperparameters = grid_GBR.best_params_
        elif self.model_type == "SVM":
            self.model = svm.SVR()
            parameters = {"kernel": ["linear", "poly", "rbf", "sigmoid"],
                         "degree": range(1, 6)}
            grid_GBR = GridSearchCV(estimator=self.model, param_grid = parameters, cv = 5, n_jobs=-1, scoring=self.scoring)
            self.pipeline = StandardScaler()
            self.pipeline.fit(X)
            X = self.pipeline.transform(X)
            grid_GBR.fit(X, y)
            self.hyperparameters = grid_GBR.best_params_
        elif self.model_type == "GradientBoostingRegressor":
            self.model = GradientBoostingRegressor()
            parameters = {"learning_rate":np.logspace(-2,0,3),
                         "n_estimators":range(20, 101, 20)}
            grid_GBR = GridSearchCV(estimator=self.model, param_grid = parameters, cv = 5, n_jobs=-1, scoring=self.scoring)
            grid_GBR.fit(X, y)
            self.hyperparameters = grid_GBR.best_params_
        


class ModelSelector:
    def __init__(self, scoring, pnl_score):
        self.scoring = scoring
        self.pnl_score = pnl_score
        self.available_models = ["SARIMA", "SARIMAX", "LinearRegression", "Lasso", "Ridge"
                                 , "ElasticNet", "SVM", "GradientBoostingRegressor"]


    def select_model(self, X, y, train_index, val_index):
        best_models = {name : Model(name, self.scoring) for name in self.available_models}

        for _, best_model in tqdm(best_models.items()):
            best_model.optimize_hyperparameters(X.loc[train_index], y[train_index])
            best_model.fit(X.loc[train_index], y[train_index])

        all_scores = {name : [] for name in self.available_models}
        for date in tqdm(val_index):
            for name, best_model in best_models.items():
                prediction = best_model.predict(X.loc[date], horizon=1)
                score = self.pnl_score(y[date], prediction)
                all_scores[name].append(score)
                best_model.fit(X.loc[:date], y[:date])

        best_score, best_type, best_hyperparameters = -1, None, None
        for name, scores in tqdm(all_scores.items()):
            score = np.array(scores).mean()
            if score > best_score:
                best_type = name
                best_hyperparameters = best_models[name].hyperparameters

        return Model(model_type=best_type, hyparameters=best_hyperparameters)
