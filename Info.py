import numpy as np


class Info:
    def __init__(self):
        self.LoadData()

    def LoadData(self):
        self.data = np.genfromtxt("Train.txt", delimiter=",")
        self.label = self.data[:, -1].T
        self.data = self.data[:, :-1].T
        self.test = np.genfromtxt("Test.txt", delimiter=",")
        self.testData = self.test[:, :-1].T
        self.testlable = self.test[:, -1].T
