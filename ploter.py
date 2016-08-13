import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
# from matplotlib.mlab import griddata
from scipy.interpolate import griddata

def gauss(x, a, mean, sigma):
    return a * norm.pdf(x, mean, sigma)

def gauss2d(x, y, a, x_0, y_0, sigma_x, sigma_y):

    sqdist_x = x - x_0
    sqdist_x = sqdist_x ** 2
    gauss_x = np.exp(-sqdist_x / 2. / sigma_x ** 2)
    sqdist_y = y - y_0
    sqdist_y = sqdist_y ** 2
    gauss_y = np.exp(-sqdist_y / 2. / sigma_y ** 2)

    return a / (2. * np.pi * sigma_x * sigma_y) * gauss_x * gauss_y 
    # return a * norm.pdf(x, x_0, sigma_x) * norm.pdf(y, y_0, sigma_y)


def interpol2d(X, Y, Z, n_grid_x, n_grid_y, sigma_x=None, sigma_y=None):
    # smoothing 2d data
    # from 1-d data of x, y, and z
    X, Y, Z = np.array([X, Y, Z])
    y_min, y_max = np.min(Y), np.max(Y)
    x_min, x_max = np.min(X), np.max(X)
    sigma_y = sigma_y or (y_max - y_min) / n_grid_y * (n_grid_y * 0.015)
    sigma_x = sigma_x or (x_max - x_min) / n_grid_x * (n_grid_x * 0.015)

    x_bin = np.linspace(x_min, x_max, n_grid_x)
    y_bin = np.linspace(y_min, y_max, n_grid_y)
    X_i, Y_i = np.meshgrid(x_bin, y_bin)

    index =  np.where(Z > Z.max() * 1E-2)
    # print Z.shape
    X = X[index]
    Y = Y[index]
    Z = Z[index]
    # print Z.shape

    X = X.reshape((1,1,X.size))
    Z = Z.reshape((1,1,Z.size))

    X_i = X_i[:, :, np.newaxis]
    Y = Y.reshape((1,1,Y.size))
    Y_i = Y_i[:, :, np.newaxis]

    temp_data = gauss2d(X_i, Y_i, Z, X, Y, sigma_x, sigma_y)
    data_interpol = np.sum(temp_data, axis=2)
    return np.rot90(data_interpol, 3)


def grid(x, y, z, res_x=100, res_y=100):
    "Convert 3 column data to matplotlib grid"
    xi = np.linspace(min(x), max(x), res_x)
    yi = np.linspace(min(y), max(y), res_y)
    # np.meshgrid
    X, Y = np.meshgrid(xi, yi)
    # Z = griddata(x, y, z, xi, yi)
    Z = griddata((x, y), z, (X, Y), method='cubic', fill_value=0)
    return X, Y, Z

if __name__ == '__main__':
    x = np.random.rand(500) * 10# * np.pi
    y = np.random.rand(500) * 10# * np.pi
    z = (x - 5) ** 2 + (y - 5) ** 2 + np.random.rand() * 10 - 5
    z = 1. / z
    # x_i, y_i, z_i = grid(x, y, z, 200, 200)
    z_i = interpol2d(x, y, z, 100, 200)

    plt.scatter(x, y, s=z*100)
    plt.imshow(z_i.T, extent=(x.min(), x.max(), y.min(), y.max()))
    plt.show()