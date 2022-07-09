import numpy as np


class Info:
    def __init__(self):
        self.LoadData()
        self.Accoracy = 0
        self.err = 0

    def LoadData(self):
        self.data = np.genfromtxt("Train.txt", delimiter=",")
        self.label = self.data[:, -1].T
        self.data = self.data[:, :-1].T
        self.test = np.genfromtxt("Test.txt", delimiter=",")
        self.testData = self.test[:, :-1].T
        self.testlable = self.test[:, -1].T

    def CheckAccuracy(self, correct_assign):
        self.Accoracy = correct_assign / len(self.testlable)

    def CalculateErrorRate(self):
        self.err = 1 - self.Accoracy

    def ValidatClassfier(self, sum_correct_assign, classfierName):
        self.CheckAccuracy(sum_correct_assign)
        self.CalculateErrorRate()
        print(classfierName + ':  Error rate %f%%' % (self.err * 100))
