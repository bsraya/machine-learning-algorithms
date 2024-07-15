from sklearn.datasets import load_iris
from Regressions import LogisticRegression

iris = load_iris()

X, y = iris.data, iris.target

linreg = LogisticRegression(epochs = 1000, learning_rate=0.01)

linreg.fit(X, y)
print(f"Intercept: {linreg.intercept_()}")
print(f"Coefficients: {linreg.coef_()}")