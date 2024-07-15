from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer, load_boston
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error

class SklearnModel:
    def __init__(self, task_type='classification', **params):
        self.task_type = task_type
        self.params = params
        
        if self.task_type == 'classification':
            self.model = LogisticRegression(**self.params)
        elif self.task_type == 'regression':
            self.model = LinearRegression(**self.params)
        else:
            raise ValueError("task_type must be either 'classification' or 'regression'")
    
    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
    
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
    
    clf_model = SklearnModel(task_type='classification', solver='liblinear')
    clf_model.fit(X_train, y_train)
    clf_model.evaluate(X_test, y_test)

    # Regression example
    data = load_boston()
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)
    
    # Standardize data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    reg_model = SklearnModel(task_type='regression')
    reg_model.fit(X_train, y_train)
    reg_model.evaluate(X_test, y_test)
