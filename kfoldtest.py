import copy
import 

from ml_from_scratch.linear_model import LinearRegression
from ml_from_scratch.metrics import mean_squared_error
from ml_from_scratch.model_selection import KFold, cross_val_score

data = pd.read_csv("data/auto.csv")

print(data.head())

from sklearn.model_selection import train_test_split

X = data.drop(columns=["mpg"])
y = data["mpg"]

print(f"X = {X} \n y - {y}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# create a model
cols_list = [
    ["displacement"],
    ["horsepower"],
    ["weight"],
    ["displacement", "horsepower"],
    ["displacement", "weight"],
    ["horsepower", "weight"],
    ["displacement", "horsepower", "weight"],
]

# cross validate model
mse_training_list = []
mse_valid_list = []

for cols in cols_list:
    X_cols = X_train[cols]
    y = y_train

    mse_train_cols, mse_test_cols = cross_val_score(
        estimator=LinearRegression(), X=X_cols, y=y, cv=5, scoring="mean_squared_error"
    )

    mse_training_list.append(np.mean(mse_train_cols))
    mse_valid_list.append(np.mean(mse_test_cols))


print(mse_training_list)

summary = pd.DataFrame(
    {
        "cols": cols_list,
        "MSE Training": mse_training_list,
        "MSE Validation": mse_valid_list,
    }
)

print(summary)

# find the best model
ind_best = summary["MSE Validation"].argmin()
col_best = summary.loc[ind_best]["cols"]
print(f"Best Model Feature: {col_best}")
print(f"Best Valid Score: {summary['MSE Validation'].min()}")

# Train the best model
print("")
print("re-train the best model")
X_best = X_train[col_best]
linreg_best = LinearRegression()
linreg_best.fit(X_best, y_train)

# predict the test data
X_test_best = X_test[col_best]
y_pred_test = linreg_best.predict(X_test_best)

# Calculate MSE
mse_best = mean_squared_error(y_test, y_pred_test)
print(f"MSE best model : {mse_best}")
