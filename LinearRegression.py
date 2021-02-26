from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt

diabetes = datasets.load_diabetes()

# print(diabetes)
# print(diabetes.DESCR)
print(diabetes.data)
print(diabetes.target)

x = diabetes.data
y = diabetes.target
# print(x.shape,y.shape)

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2)
# print(X_train.shape, Y_train.shape)
# print(X_test.shape, Y_test.shape)

model = linear_model.LinearRegression()
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)

print("Coefficients: ", model.coef_)
print("Intercept: ", model.intercept_)
print("Mean Squared Error (MSE): %.2f" % mean_squared_error(Y_test, Y_pred))
print("Coefficient of Determination (R^2): %.2f" % r2_score(Y_test, Y_pred))

print("Y_test expected: ", Y_test)
print("Y predicted values: ", Y_pred)

sns.scatterplot(x=Y_test, y=Y_pred)
plt.show()
