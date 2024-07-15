import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer, load_boston
import numpy as np

class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, task_type='classification'):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.task_type = task_type
        
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
        if self.task_type == 'classification':
            self.activation = nn.Sigmoid()
        elif self.task_type == 'regression':
            self.activation = nn.Identity()
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        out = self.activation(out)
        return out

class PyTorchRNN:
    def __init__(self, task_type='classification', input_size=10, hidden_size=50, num_layers=2, learning_rate=0.001):
        self.task_type = task_type
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.output_size = 1
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = RNNModel(input_size, hidden_size, num_layers, self.output_size, task_type).to(self.device)
        self.criterion = nn.BCELoss() if task_type == 'classification' else nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
    
    def fit(self, X_train, y_train, epochs=10, batch_size=32):
        self.model.train()
        X_train, y_train = torch.tensor(X_train, dtype=torch.float32).to(self.device), torch.tensor(y_train, dtype=torch.float32).to(self.device)
        dataset = torch.utils.data.TensorDataset(X_train, y_train)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        for epoch in range(epochs):
            for inputs, targets in loader:
                outputs = self.model(inputs)
                loss = self.criterion(outputs.squeeze(), targets)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
    
    def predict(self, X):
        self.model.eval()
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            outputs = self.model(X).squeeze()
        return outputs.cpu().numpy()
    
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
    
    # Reshape data for RNN
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    
    clf_model = PyTorchRNN(task_type='classification', input_size=X_train.shape[1], hidden_size=50, num_layers=2, learning_rate=0.001)
    clf_model.fit(X_train, y_train, epochs=10, batch_size=32)
    clf_model.evaluate(X_test, y_test)

    # Regression example
    data = load_boston()
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)
    
    # Standardize data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Reshape data for RNN
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    
    reg_model = PyTorchRNN(task_type='regression', input_size=X_train.shape[1], hidden_size=50, num_layers=2, learning_rate=0.001)
    reg_model.fit(X_train, y_train, epochs=10, batch_size=32)
    reg_model.evaluate(X_test, y_test)
