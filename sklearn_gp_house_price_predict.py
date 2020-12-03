import pandas as pd
from sklearn import metrics

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from time_helper import log_time_cost


def get_train_and_test_data():
    data = pd.read_csv("./kc_house_data.csv")
    data.dropna(inplace=True)
    data = data[:500]
    data = data[(data['bedrooms'] < 10) & (data['bathrooms'] < 8)]
    x_df = data[['bedrooms', 'bathrooms', 'sqft_living', 'sqft_above', 'grade']]
    x_df = pd.get_dummies(x_df, columns=['grade'])
    y = data['price']
    return train_test_split(x_df, y, train_size=0.8, random_state=42)


@log_time_cost
def test_linear_model(x_train,x_test,y_train,y_test):
    reg = LinearRegression()
    reg.fit(x_train, y_train)
    print({
        "model": "linear_model",
        "train_r2": metrics.r2_score(y_train, reg.predict(x_train)),
        "test_r2": metrics.r2_score(y_test, reg.predict(x_test))
    })


# with calibration range
def get_kernel(features_n):
    return C(1.0, (1e-3, 1e3)) * RBF([1] * features_n, (1e-2, 1e2)) + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e+1))


@log_time_cost
def test_gp(x_train,x_test,y_train,y_test):
    gp = GaussianProcessRegressor(kernel=get_kernel(features_n=len(x_train[0])), n_restarts_optimizer=10)
    gp.fit(x_train, y_train)

    print({
        "model.kernel_": gp.kernel_,
        "model.kernel_.bounds(log-scale)": gp.kernel_.bounds,
        "model.kernel_.theta(log-scale)": gp.kernel_.theta
    })

    y_train_pred = gp.predict(x_train)
    y_test_pred = gp.predict(x_test)

    print({
        "model": "GaussianProcessRegressor",
        "train_r2": metrics.r2_score(y_train, y_train_pred),
        "test_r2": metrics.r2_score(y_test, y_test_pred)
    })


def demo():
    x_train,x_test,y_train,y_test = get_train_and_test_data()
    # dataframe -> array
    x_train, x_test, y_train, y_test = x_train.values, x_test.values, y_train.values, y_test.values

    print({
        "x_train": x_train[:2],
        "x_test": x_test[:2],
        "y_train": y_train[:2],
        "y_test": y_test[:2],
    })

    test_linear_model(x_train, x_test, y_train, y_test)
    test_gp(x_train, x_test, y_train, y_test)


# start the demo
demo()
