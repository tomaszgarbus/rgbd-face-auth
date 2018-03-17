import numpy as np
import math

def rotate_x(X, Y, Z, theta):
    """ Rotates coordinates X, Y, Z by angle theta around axis X """
    """
    [1           0            0]   [ X ]   [nX]
    [0  cos(theta)  -sin(theta)] * [ Y ] = [nY]
    [0  sin(theta)   cos(theta)]   [ Z ]   [nZ]
    """
    nX = X
    nY = math.cos(theta) * Y - math.sin(theta) * Z
    nZ = math.sin(theta) * Y + math.cos(theta) * Z
    return np.array([nX, nY, nZ])

def rotate_y(X, Y, Z, theta):
    """ Rotates coordinates X, Y, Z by angle theta around axis Y """
    """
    [cos(theta)   0  sin(theta)]   [ X ]   [nX]
    [0            1           0] * [ Y ] = [nY]
    [-sin(theta)  0  cos(theta)]   [ Z ]   [nZ]
    """
    nX = math.cos(theta) * X + math.sin(theta) * Z
    nY = Y
    nZ = -math.sin(theta) * X + math.cos(theta) * Z
    return np.array([nX, nY, nZ])

def rotate_z(X, Y, Z, theta):
    """ Rotates coordinates X, Y, Z by angle theta around axis Z """
    """
    [cos(theta)  -sin(theta)  0]   [ X ]   [nX]
    [sin(theta)   cos(theta)  0] * [ Y ] = [nY]
    [0            0           1]   [ Z ]   [nZ]
    """
    nX = math.cos(theta) * X - math.sin(theta) * Y
    nY = math.sin(theta) * X - math.cos(theta) * Y
    nZ = Z
    return np.array([nX, nY, nZ])