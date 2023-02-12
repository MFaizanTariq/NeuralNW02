import numpy as np
from numpy import pi

def train_data():
    n1 = 500
    n2 = 100

    r1 = np.random.rand(n1) + 1
    r2 = np.random.rand(n2)

    ang1 = np.random.rand(n1) * -pi
    ang3 = np.random.rand(n2) * pi
    ang2 = np.random.rand(n1) * pi
    ang4 = np.random.rand(n2) * -pi

    data_c1a = np.array([np.sin(ang1) * r1, np.cos(ang1) * r1]).T
    data_c1b = np.array([np.sin(ang3) * r2, np.cos(ang3) * r2 - 1]).T

    data_c2a = np.array([np.sin(ang2) * r1, np.cos(ang2) * r1 - 1]).T
    data_c2b = np.array([np.sin(ang4) * r2, np.cos(ang4) * r2]).T

    data_c1 = np.append(data_c1a, data_c1b, axis=0)
    data_c2 = np.append(data_c2a, data_c2b, axis=0)
    data = np.append(data_c1, data_c2, axis=0)

    return data


def train_data2():
    n1 = 225
    n2 = 25

    r1 = np.random.rand(n1) + 1
    r2 = np.random.rand(n2)

    ang1 = np.random.rand(n1) * -pi
    ang3 = np.random.rand(n2) * pi
    ang2 = np.random.rand(n1) * pi
    ang4 = np.random.rand(n2) * -pi

    data_c1a = np.array([np.sin(ang1) * r1, np.cos(ang1) * r1]).T
    data_c1b = np.array([np.sin(ang3) * r2, np.cos(ang3) * r2 - 1]).T

    data_c2a = np.array([np.sin(ang2) * r1, np.cos(ang2) * r1 - 1]).T
    data_c2b = np.array([np.sin(ang4) * r2, np.cos(ang4) * r2]).T

    data_c1 = np.append(data_c1a, data_c1b, axis=0)
    data_c2 = np.append(data_c2a, data_c2b, axis=0)
    data = np.append(data_c1, data_c2, axis=0)

    return data