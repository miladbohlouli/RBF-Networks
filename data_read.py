import os
import numpy as np
from matplotlib import pyplot as plt
from RBF_ANN import RBF_NN
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

np.set_printoptions(precision=2, threshold=20)

def read_data(path = 'data', draw=False):
    data = np.loadtxt(os.path.join(path, "data1.txt"), delimiter='\t')
    if draw:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for clus in np.unique(data[:, 3]):
            data_clus = data[data[:, 3] == clus]
            ax.scatter(data_clus[:, 0], data_clus[:, 1], data_clus[:, 2], s=3)
        plt.show()
    return data[:, 0:3], data[:, 3]


###########################################################################
#   Takes the list of the models and saves them in the given path
#       number of the models could be one or more
###########################################################################
def save_models(models, path):
    for i in range(len(models)):
        models[i].save(path)


###########################################################################
#   Takes the path of the saved model and loads it
###########################################################################
def load_models(models_path):
    models = []
    for i in range(len(models_path)):
        model = RBF_NN()
        model.load(models_path[i])
        models.append(model)
    return models

def generate_regression_data(part='1', draw=False, num_data=500, variance=1):
    if part == '1':
        data_x = np.linspace(-540, 540, num_data)
        data_y = np.sin(data_x * np.pi / 180)
        if draw:
            plt.scatter(data_x, data_y, s=3)
            plt.show()
        return data_x, data_y

    elif part == '2':
        data_x = np.linspace(-540, 540, num_data)
        data_y = np.sin(data_x * np.pi / 180) + np.random.normal(0, variance, num_data)

        if draw:
            plt.scatter(data_x, data_y, s=3)
            plt.show()
        return data_x, data_y

    elif part == '3':
        file = pd.read_csv("data/part2_data.csv").to_numpy()
        data_x = np.arange(365)
        data_y = file[:-1, 1]

        if draw:
            plt.scatter(data_x, data_y, s=3)
            plt.show()
        return data_x, data_y


def MSE(predictions, true_value):
    return np.mean(np.power(predictions.ravel() - true_value.ravel(), 2))

