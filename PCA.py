import numpy as np
import pylab


class PCA:

    def __init__(self, m):
        # we want the lorgest m egin venctor
        self.m = m
        self.egin_vector = []
        self.mean = 0
        self.convertion = 0
        self.data = []
        self.LoadData()
        self.CalculateMean()
        self.CenterData()
        self.ConvertionMatrix()
        self.Eigenvectors()

    def LoadData(self):
        self.data = np.genfromtxt("test.txt", delimiter=",")


    def VectorCol(self, data):
        return data.reshape((data.size, 1))

    def VectorRow(self, data):
        return data.reshape((1, data.size))

    def CalculateMean(self):
        self.mean = self.data.mean(1)

    def CenterData(self):
        self.data = self.data - self.VectorCol(self.mean)

    def ConvertionMatrix(self):
        self.convertion = np.dot(self.data, self.data.T) / self.data.shape[1]

    def Eigenvectors(self):
        self.egin_vector, s, vh = np.linalg.svd(self.convertion)
        self.egin_vector = self.egin_vector[:, 0:self.m]

    def PlotFunction(self):
        projection_list = np.dot(self.egin_vector.T, self.data)
        pylab.scatter(projection_list[0], projection_list[1])
        pylab.show()


hi = PCA(3)
hi.PlotFunction()
