from numpy import zeros

class LinearRegression:
    def __init__(self, learning_rate: float = 0.01, iterations: int = 1000):
        self.iterations: int = iterations
        self.learning_rate: float = learning_rate
        self.intercept: float = 0.0
        self.coefficients: list = list()

    def intercept_(self):
        return self.intercept
    
    def coef_(self):
        return self.coefficients
    
    def loss_history_(self):
        return self.loss_history
    
    # def fit(self, X, y):
    #     self.X = X
    #     self.y = y
    #     self.length = len(self.X.shape[0])
    #     self.coefficients = zeros(self.X.shape[1])

    #     for _ in range(self.iterations):
    #         predictions = self.predict(self.X)
    #         self.intercept, self.coefficients = SGD(self.X, self.y, self.learning_rate)
    #         self.loss_history.append(
    #             MeanSquaredError(predictions, self.y)
    #         )

