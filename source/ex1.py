import numpy as np
import plot_data as plt
import linear_regression as lr

def load_data(file_name):
    data = []
    with open(file_name) as file:
        for line in file:
            data.append(line.replace("\n", "").split(","))
    data = np.array(data, dtype=np.float)
    return data

data = load_data("exdata1.txt")

def get_training_examples(data):
    dt_size = np.shape(data)    # get data's shape

    # convert 1D array to 2D array
    X = np.reshape(data[:, :dt_size[1]-1], (dt_size[0], dt_size[1]-1))
    y = np.reshape(data[:, -1], (dt_size[0], 1))
    return X, y

X, y = get_training_examples(data)
X = lr.feature_normalize(X)

plt.plot_data(X, y, x_label="Population", y_label="Revenue")

X = np.insert(X, 0, 1, axis=1)
theta = np.zeros((np.shape(X)[1] , 1))

alpha = 0.5
iterations = 1200

theta = lr.gradient_descend(X, y, theta, alpha, iterations)
theta1 = lr.normal_equation(X, y)

plt.plot_data(X[:, 1], X.dot(theta), hold=True, dis_type="b-")

print(theta)
print(theta1)


