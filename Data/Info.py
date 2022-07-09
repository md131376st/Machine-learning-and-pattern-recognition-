import os

import numpy as np


class Info:
    def __init__(self, data=None, test=None, isKfold=False):
        if not isKfold:
            self.LoadData()
        else:
            self.data = data
            self.test = test
        self.label = self.data[:, -1].T
        self.data = self.data[:, :-1].T

        self.testData = self.test[:, :-1].T
        self.testlable = self.test[:, -1].T

        self.Accoracy = 0
        self.err = 0

    def LoadData(self):
        self.data = np.genfromtxt(os.path.join(os.path.dirname(os.path.abspath(__file__))) + "/Train.txt",
                                  delimiter=",")
        self.test = np.genfromtxt(os.path.join(os.path.dirname(os.path.abspath(__file__))) + "/Test.txt", delimiter=",")

    def TransferData(self):
        pass

    def CheckAccuracy(self, correct_assign):
        self.Accoracy = correct_assign / len(self.testlable)

    def CalculateErrorRate(self):
        self.err = 1 - self.Accoracy

    def ValidatClassfier(self, sum_correct_assign, classfierName):
        self.CheckAccuracy(sum_correct_assign)
        self.CalculateErrorRate()
        print(classfierName + ':  Error rate %f%%' % (self.err * 100))


class KFold:
    def __init__(self, k):
        self.k = k
        self.foldList = []
        self.LoadData()
        self.infoSet = []
        self.GenerateInfoDataWithTest()
        pass

    def LoadData(self):
        self.data = np.genfromtxt(os.path.join(os.path.dirname(os.path.abspath(__file__)))
                                  + "/Train.txt",
                                  delimiter=",")
        self.size = int((self.data.shape[0]) / 2)
        self.wemonData = self.data[self.size:, :]
        self.MenData = self.data[:self.size, :]
        self.foldsize = int(self.size / self.k)

        for i in range(self.k):
            self.foldList.append(np.concatenate((self.MenData[i * self.foldsize:self.foldsize * (i + 1), :],
                                                 self.wemonData[i * self.foldsize:self.foldsize * (i + 1), :])))
            pass

    def GenerateInfoDataWithTest(self):
        for i in range(self.k):

            test = self.foldList[i]
            data = np.zeros(shape=(0, 13))
            for j in range(i):
                data = np.concatenate((data, self.foldList[i]))
            for j in range(i + 1, self.k):
                data = np.concatenate((data, self.foldList[j]))
            self.infoSet.append(Info(data, test, True))
