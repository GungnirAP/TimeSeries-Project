import pandas as pd

class Model:
    def __init__(self, 
                 model_type="ARIMA", 
                #  *model_args, 
                #  **model_kwargs
                 ):
        self.model_type = model_type
        self.pipeline = None
        # if model_type == "":
        #     self.pipeline = ...
        #     self.hyperparameters = None

    def predict(self, series, horizon=1):
        prediction = pd.Series() # поставить индекс!
        return prediction
    
    def fit(self, X, y):
        # fit self.pipeline wrt self.hyperparameters
        return self.pipeline

class ModelSelector:
    def __init__(self):
        pass

    def select_model(self, X, y):
        return Model()