import os

import numpy as np


class Curve:
    """an object to read curves which created by 3-Matic"""
    def __init__(self, path: str) -> None:
        assert os.path.exists(path), f"given path dont exists, {path}"
        self.path = path
    
    def curve(self):
        file = open(self.path)
        curve = []
        for line in file:
            if not line.startswith('%'):
                x, y, z, _ = line.split(' ')
                x = float(x)
                y = float(y)
                z = float(z)
                curve.append([x, y, z])
        self.coor = curve
        return self
    
    def to_list(self):
        return self.coor
    
    def to_numpy(self):
        return np.array(self.coor, np.float32)