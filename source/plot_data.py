import matplotlib.pyplot as plt
import numpy as np

def load_data(file_name):
    data = []
    with open(file_name) as file:
        for line in file:
            data.append(line.replace("\n", "").split(","))
    data = np.array(data, dtype=np.float)
    return data

def get_training_examples(data):
    dt_size = np.shape(data)    # get data's shape

    # convert 1D array to 2D array
    X = np.reshape(data[:, :dt_size[1]-1], (dt_size[0], dt_size[1]-1))
    y = np.reshape(data[:, -1], (dt_size[0], 1))
    return X, y

def plot_data(x, y, dis_type = "rx", hold=False, x_label = None, y_label = None):
    plt.plot(x, y, dis_type)
    plt.xlabel(xlabel=x_label)
    plt.ylabel(ylabel=y_label)
    if not hold:
        plt.show()