import lightgbm as lgb
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer, load_boston

class LightGBMModel:
    def __init__(self, task_type='classification', **params):
        self.task_type = task_type
        self.params = params
        
        if self.task_type == 'classification':
            self.model = lgb.LGBMClassifier(**self.params)
        elif self.task_type == 'regression':
            self.model = lgb.LGBMRegressor(**self.params)
        else:
            raise ValueError("task_type must be either 'classification' or 'regression'")
    
    def fit(self, X_train, y_train, eval_set=None, verbose=False):
        self.model.fit(X_train, y_train, eval_set=eval_set, verbose=verbose)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def evaluate(self, X_test, y_test):
        predictions = self.predict(X_test)
        
        if self.task_type == 'classification':
            score = accuracy_score(y_test, predictions)
            print(f"Accuracy: {score}")
        elif self.task_type == 'regression':
            score = mean_squared_error(y_test, predictions, squared=False)
            print(f"RMSE: {score}")
        
        return score

# Example usage:
if __name__ == "__main__":
    # Classification example
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)
    
    clf_model = LightGBMModel(task_type='classification', n_estimators=100, learning_rate=0.1, max_depth=6)
    clf_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=10)
    clf_model.evaluate(X_test, y_test)

    # Regression example
    data = load_boston()
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)
    
    reg_model = LightGBMModel(task_type='regression', n_estimators=100, learning_rate=0.1, max_depth=6)
    reg_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=10)
    reg_model.evaluate(X_test, y_test)
