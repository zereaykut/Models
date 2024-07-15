import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer, load_boston

class LSTMModel:
    def __init__(self, task_type='classification', input_shape=(10, 1), units=50, learning_rate=0.001):
        self.task_type = task_type
        self.input_shape = input_shape
        self.units = units
        self.learning_rate = learning_rate
        
        if self.task_type not in ['classification', 'regression']:
            raise ValueError("task_type must be either 'classification' or 'regression'")
        
        self.model = self._build_model()
    
    def _build_model(self):
        model = Sequential()
        model.add(LSTM(self.units, input_shape=self.input_shape, activation='relu'))
        if self.task_type == 'classification':
            model.add(Dense(1, activation='sigmoid'))
            model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
        elif self.task_type == 'regression':
            model.add(Dense(1))
            model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mse', metrics=['mse'])
        
        return model
    
    def fit(self, X_train, y_train, epochs=10, batch_size=32, validation_data=None):
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=validation_data)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def evaluate(self, X_test, y_test):
        predictions = self.predict(X_test)
        if self.task_type == 'classification':
            predictions = (predictions > 0.5).astype(int)
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
    
    # Reshape data for LSTM
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    
    clf_model = LSTMModel(task_type='classification', input_shape=(X_train.shape[1], 1), units=50)
    clf_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
    clf_model.evaluate(X_test, y_test)

    # Regression example
    data = load_boston()
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)
    
    # Standardize data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Reshape data for LSTM
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    
    reg_model = LSTMModel(task_type='regression', input_shape=(X_train.shape[1], 1), units=50)
    reg_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
    reg_model.evaluate(X_test, y_test)
