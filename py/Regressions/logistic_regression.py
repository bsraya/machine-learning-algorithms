import numpy as np

class LogisticRegression:
    def __init__(self, epochs: int = 100, learning_rate: float = 0.01):
        self.iterations: int = 1_000
        self.intercept: float = 0.0
        self.coefficients: list = list()
        self.optimizer: str = "sgd"
        self.loss_history: list = list()
        self.epochs: int = epochs
        self.learning_rate: float = learning_rate

    def intercept_(self):
        return self.intercept
    
    def coef_(self):
        return self.coefficients.flatten()
    
    def loss_history_(self):
        return self.loss_history
    
    def sigmoid(self, x: float):
        return 1 / (1 + np.exp(-x))
    
    def linear_function(self, intercept: float, coefficients: np.array, x):
        return intercept + np.dot(coefficients, x.T)
    
    def threshold(self, x):
        return np.where(x > 0.5, 1, 0)
    
    def log_loss(self, y, y_pred):
        return np.mean(
            y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred)
        )

    def fit(self, X, y):
        length = len(y)
        self.coefficients = np.zeros((1, X.shape[1]))

        for i in range(self.epochs):
            y_pred = self.sigmoid(self.linear_function(
                self.intercept,
                self.coefficients,
                X
            ).flatten())

            error = (y_pred - y).reshape(-1, 1)

            self.intercept = self.intercept - self.learning_rate * sum(error) / length
            self.coefficients = self.coefficients - self.learning_rate * sum(np.dot(error.T, X)) / length