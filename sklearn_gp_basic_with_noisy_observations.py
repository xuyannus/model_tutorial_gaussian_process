import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from matplotlib import pyplot as plt


def target_f(x):
    return x * np.sin(x)


def get_train_data(size=20):
    X = np.atleast_2d(np.linspace(0.1, 10, size)).T
    y = target_f(X).ravel()

    # add independent Normal noise
    dy = 0.5 + 1.0 * np.random.random(y.shape)
    noise = np.random.normal(0, dy)
    return X, y + noise


# with calibration range
def get_kernel():
    return C(1.0, (1e-3, 1e3)) * RBF(1, (1e-2, 1e2)) + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e+1))


def build_gp_model(X, y):
    gp = GaussianProcessRegressor(kernel=get_kernel(), n_restarts_optimizer=10)
    gp.fit(X, y)

    #https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html#sklearn.gaussian_process.GaussianProcessRegressor.score
    print("score:", gp.score(X, y))
    return gp


def describe_gp(model):
    print({
        "model.kernel_": model.kernel_,
        "model.kernel_.bounds(log-scale)": model.kernel_.bounds,
        "model.kernel_.theta(log-scale)": model.kernel_.theta
    })


def plot_gp(x, y, y_pred, sigma):
    plt.figure()
    plt.plot(x, y, 'r:', label=r'$f(x) = x\,\sin(x)$')
    plt.plot(x, y_pred, 'b-', label='Prediction')
    plt.fill(np.concatenate([x, x[::-1]]),
             np.concatenate([y_pred - 1.9600 * sigma,
                             (y_pred + 1.9600 * sigma)[::-1]]),
             alpha=.5, fc='b', ec='None', label='95% confidence interval')
    plt.xlabel('$x$')
    plt.ylabel('$f(x)$')
    plt.ylim(-10, 20)
    plt.legend(loc='upper left')
    plt.show()


def demo():
    N = 100
    X, y = get_train_data(N)
    gp = build_gp_model(X, y)
    describe_gp(gp)

    test_x = np.atleast_2d(np.linspace(0, 10, 1000)).T
    y_pred, sigma = gp.predict(test_x, return_std=True)
    plot_gp(test_x, target_f(test_x).ravel(), y_pred, sigma)


# start the demo
demo()
