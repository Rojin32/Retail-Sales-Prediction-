import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
df=pd.read_csv('business.retailsales2.csv')
print(df.head())
print(df.columns)
X = df.drop(columns=["Month", "Total Sales", "Net Sales", "Shipping"])
y = df["Total Sales"]
print("Features used for prediction:", X.columns)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared error",mse)
print("R2 score",r2)
print(model.coef_)
# Sample data for testing
sample_data = [[2019, 80, 10000, -200, -500]]
prediction = model.predict(sample_data)
print("Predicted Total Sales:", prediction)
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y.min(), y.max()], [y.min(), y.max()], color="red", linestyle="--")
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.show()
